// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess.h"

#include <cmath>
#include <iterator>
#include <math.h>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"
#include <android/log.h>

#define TAG "AEKE_AI_LOG"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

static char *labels[OBJ_CLASS_NUM];

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

inline static int clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}

char *readLine(FILE *fp, char *buffer, int *len) {
  int ch;
  int i = 0;
  size_t buff_len = 0;

  buffer = (char *)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void *tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char *)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char *fileName, char *lines[], int max_line) {
  FILE *file = fopen(fileName, "r");
  char *s;
  int i = 0;
  int n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

int loadLabelName(const char *locationFilename, char *label[]) {
  // printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                              float ymax0, float xmin1, float ymin1,
                              float xmax1, float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations,
               std::vector<int> classIds, std::vector<int> &order, int filterId,
               float threshold) {
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left,
                                     int right, std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

static int process(int8_t *input, int *anchor, int grid_h, int grid_w,
                   int height, int width, int stride, std::vector<float> &boxes,
                   std::vector<float> &objProbs, std::vector<int> &classId,
                   float threshold, int32_t zp, float scale) {
  int validCount = 0;
  int grid_len = grid_h * grid_w; // 20 40 80
  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
  // PROP_BOX_SIZE:85
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence =
            input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w +
                  j]; // (85 * a + 4)*20/40/80**2 + i * 20/40/80 + j

        if (box_confidence >= thres_i8) {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset;
          float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y =
              (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w =
              (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h =
              (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres_i8) {
            objProbs.push_back(
                (deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                (deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h,
                 int model_in_w, float conf_threshold, float nms_threshold,
                 BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group) {
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h,
                        model_in_w, stride0, filterBoxes, objProbs, classId,
                        conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h,
                        model_in_w, stride1, filterBoxes, objProbs, classId,
                        conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h,
                        model_in_w, stride2, filterBoxes, objProbs, classId,
                        conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - pads.left;
    float y1 = filterBoxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left =
        (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top =
        (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right =
        (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom =
        (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop = obj_conf;
    char *label = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i,
    // group->results[last_count].box.left, group->results[last_count].box.top,
    //        group->results[last_count].box.right,
    //        group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void deinitPostProcess() {
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x) {
  // return exp(x);
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672 * x + 1064807160.56887296);
  return v.f;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1,
                        float XMin2, float YMin2, float XMax2, float YMax2) {
  float Inter = 0;
  float Total = 0;
  float XMin = 0;
  float YMin = 0;
  float XMax = 0;
  float YMax = 0;
  float Area1 = 0;
  float Area2 = 0;
  float InterWidth = 0;
  float InterHeight = 0;

  XMin = ZQ_MAX(XMin1, XMin2);
  YMin = ZQ_MAX(YMin1, YMin2);
  XMax = ZQ_MIN(XMax1, XMax2);
  YMax = ZQ_MIN(YMax1, YMax2);

  InterWidth = XMax - XMin;
  InterHeight = YMax - YMin;

  InterWidth = (InterWidth >= 0) ? InterWidth : 0;
  InterHeight = (InterHeight >= 0) ? InterHeight : 0;

  Inter = InterWidth * InterHeight;

  Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
  Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

  Total = Area1 + Area2 - Inter;

  return float(Inter) / float(Total);
}

static float DeQnt2F32(int8_t qnt, int zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

/****** yolov8 ****/
GetResultRectYolov8::GetResultRectYolov8() {}

GetResultRectYolov8::~GetResultRectYolov8() {}

float GetResultRectYolov8::sigmoid(float x) { return 1 / (1 + fast_exp(-x)); }

int GetResultRectYolov8::GenerateMeshgrid() {
  int ret = 0;
  if (headNum == 0) {
    printf("=== yolov8 Meshgrid  Generate failed! \n");
  }

  for (int index = 0; index < headNum; index++) {
    for (int i = 0; i < mapSize[index][0]; i++) {
      for (int j = 0; j < mapSize[index][1]; j++) {
        meshgrid.push_back(float(j + 0.5));
        meshgrid.push_back(float(i + 0.5));
      }
    }
  }

  // printf("=== yolov8 Meshgrid  Generate success! \n");

  return ret;
}

int GetResultRectYolov8::GetConvDetectionResult(
    int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale,
    std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints,
    int img_width, int img_height) {
  int ret = 0;
  if (meshgrid.empty()) {
    ret = GenerateMeshgrid();
  }
  if (!DetectiontRects.empty() || !DetectKeyPoints.empty()) {
    DetectiontRects.clear();
    DetectKeyPoints.clear();
  }

  int gridIndex = -2;
  float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
  float cls_val = 0;
  float cls_max = 0;
  int cls_index = 0;

  int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_pose = 0;
  float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_pose = 0;
  yoloKeyPoint Point;

  std::vector<DetectRect> detectRects;

  for (int index = 0; index < headNum; index++) {
    int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
    int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
    int8_t *pose = (int8_t *)pBlob[index + headNum * 2];

    quant_zp_reg = qnt_zp[index * 2 + 0];
    quant_zp_cls = qnt_zp[index * 2 + 1];
    quant_zp_pose = qnt_zp[index + headNum * 2];

    quant_scale_reg = qnt_scale[index * 2 + 0];
    quant_scale_cls = qnt_scale[index * 2 + 1];
    quant_scale_pose = qnt_scale[index + headNum * 2];

    for (int h = 0; h < mapSize[index][0]; h++) {
      for (int w = 0; w < mapSize[index][1]; w++) {
        gridIndex += 2;

        for (int cl = 0; cl < class_num; cl++) {
          cls_val =
              sigmoid(DeQnt2F32(cls[cl * mapSize[index][0] * mapSize[index][1] +
                                    h * mapSize[index][1] + w],
                                quant_zp_cls, quant_scale_cls));

          if (0 == cl) {
            cls_max = cls_val;
            cls_index = cl;
          } else {
            if (cls_val > cls_max) {
              cls_max = cls_val;
              cls_index = cl;
            }
          }
        }

        if (cls_max > objectThresh) {
          xmin = (meshgrid[gridIndex + 0] -
                  DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] +
                                h * mapSize[index][1] + w],
                            quant_zp_reg, quant_scale_reg)) *
                 strides[index];
          ymin = (meshgrid[gridIndex + 1] -
                  DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] +
                                h * mapSize[index][1] + w],
                            quant_zp_reg, quant_scale_reg)) *
                 strides[index];
          xmax = (meshgrid[gridIndex + 0] +
                  DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] +
                                h * mapSize[index][1] + w],
                            quant_zp_reg, quant_scale_reg)) *
                 strides[index];
          ymax = (meshgrid[gridIndex + 1] +
                  DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] +
                                h * mapSize[index][1] + w],
                            quant_zp_reg, quant_scale_reg)) *
                 strides[index];

          xmin = xmin > 0 ? xmin : 0;
          ymin = ymin > 0 ? ymin : 0;
          xmax = xmax < input_w ? xmax : input_w;
          ymax = ymax < input_h ? ymax : input_h;

          if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h) {
            DetectRect temp;
            temp.xmin = xmin / input_w;
            temp.ymin = ymin / input_h;
            temp.xmax = xmax / input_w;
            temp.ymax = ymax / input_h;
            temp.classId = cls_index;
            temp.score = cls_max;

            for (int kc = 0; kc < keypoint_num; kc++) {
              Point.x = (DeQnt2F32(pose[(kc * 3 + 0) * mapSize[index][0] *
                                            mapSize[index][1] +
                                        h * mapSize[index][1] + w],
                                   quant_zp_pose, quant_scale_pose) *
                             2 +
                         (meshgrid[gridIndex + 0] - 0.5)) *
                        strides[index] / input_w;
              Point.y = (DeQnt2F32(pose[(kc * 3 + 1) * mapSize[index][0] *
                                            mapSize[index][1] +
                                        h * mapSize[index][1] + w],
                                   quant_zp_pose, quant_scale_pose) *
                             2 +
                         (meshgrid[gridIndex + 1] - 0.5)) *
                        strides[index] / input_h;
              Point.score = sigmoid(DeQnt2F32(
                  pose[(kc * 3 + 2) * mapSize[index][0] * mapSize[index][1] +
                       h * mapSize[index][1] + w],
                  quant_zp_pose, quant_scale_pose));

              temp.keyPoints.push_back(Point);
            }

            detectRects.push_back(temp);
          }
        }
      }
    }
  }

  std::sort(detectRects.begin(), detectRects.end(),
            [](DetectRect &Rect1, DetectRect &Rect2) -> bool {
              return (Rect1.score > Rect2.score);
            });

  for (int i = 0; i < detectRects.size(); ++i) {
    float xmin1 = detectRects[i].xmin;
    float ymin1 = detectRects[i].ymin;
    float xmax1 = detectRects[i].xmax;
    float ymax1 = detectRects[i].ymax;
    int classId = detectRects[i].classId;
    float score = detectRects[i].score;

    if (classId != -1) {
      // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1的格式存放在vector<float>中
      DetectiontRects.push_back(float(classId));
      DetectiontRects.push_back(float(score));
      DetectiontRects.push_back(float(xmin1) * float(img_width) + 0.5);
      DetectiontRects.push_back(float(ymin1) * float(img_height) + 0.5);
      DetectiontRects.push_back(float(xmax1) * float(img_width) + 0.5);
      DetectiontRects.push_back(float(ymax1) * float(img_height) + 0.5);

      // 每个检测框对应的17个关键点按照（score, x, y）格式存在vector<float>中
      for (int kn = 0; kn < keypoint_num; kn++) {
        DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].score));
        DetectKeyPoints.push_back(
            float(detectRects[i].keyPoints[kn].x) * float(img_width) + 0.5);
        DetectKeyPoints.push_back(
            float(detectRects[i].keyPoints[kn].y) * float(img_height) + 0.5);
      }

      for (int j = i + 1; j < detectRects.size(); ++j) {
        float xmin2 = detectRects[j].xmin;
        float ymin2 = detectRects[j].ymin;
        float xmax2 = detectRects[j].xmax;
        float ymax2 = detectRects[j].ymax;
        float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
        if (iou > nmsThresh) {
          detectRects[j].classId = -1;
        }
      }
    }
  }

  return ret;
}

// 后处理/Post-processing
// detect_result_group_t detect_result_group;
// std::vector<float> out_scales;
// std::vector<int32_t> out_zps;
// for (int i = 0; i < io_num.n_output; ++i) {
//   out_scales.push_back(output_attrs[i].scale);
//   out_zps.push_back(output_attrs[i].zp);
// }

// post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf,
//              (int8_t *)outputs[2].buf, img_height, img_width,
//              box_conf_threshold, nms_threshold, pads, scale_w, scale_h,
//              out_zps, out_scales, &detect_result_group);

// // 绘制框体/Draw the box
// char text[256];
// for (int i = 0; i < detect_result_group.count; i++) {
//   detect_result_t *det_result = &(detect_result_group.results[i]);
//   sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
//   // 打印预测物体的信息/Prints information about the predicted object
//   printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left,
//          det_result->box.top, det_result->box.right, det_result->box.bottom,
//          det_result->prop);
//   int x1 = det_result->box.left;
//   int y1 = det_result->box.top;
//   int x2 = det_result->box.right;
//   int y2 = det_result->box.bottom;
//   rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2),
//             cv::Scalar(256, 0, 0, 256), 3);
//   putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX,
//           0.4, cv::Scalar(255, 255, 255));
// }

// cv::imwrite("out.jpg",orig_img);

// // 后处理部分yolov8
//   std::vector<float> out_scales;
//   std::vector<int32_t> out_zps;
//   for (int i = 0; i < io_num.n_output; ++i) {
//     out_scales.push_back(output_attrs[i].scale);
//     out_zps.push_back(output_attrs[i].zp);
//   }

//   int8_t *pblob[9];
//   for (int i = 0; i < io_num.n_output; ++i) {
//     pblob[i] = (int8_t *)outputs[i].buf;
//   }

//   GetResultRectYolov8 PostProcess;

//   PostProcess.GetConvDetectionResult(pblob, out_zps, out_scales,
//                                      Info.DetectiontRects,
//                                      Info.DetectKeyPoints, img_width,
//                                      img_height);
//   printf("%d\n",Info.DetectiontRects.size());

static int process_i8(int8_t *input, int grid_h, int grid_w, int height,
                      int width, int stride, std::vector<float> &boxes,
                      std::vector<float> &objProbs, std::vector<int> &classId,
                      float threshold, int32_t zp, float scale) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

  for (int i = 0; i < grid_h; ++i) {
    for (int j = 0; j < grid_w; ++j) {

      int8_t box_confidence = input[4 * grid_len + i * grid_w + j];

      if (box_confidence >= thres_i8) {
        int offset = i * grid_w + j;
        int8_t *in_ptr = input + offset;

        int8_t maxClassProbs = in_ptr[5 * grid_len];
        int maxClassId = 0;
        for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
          int8_t prob = in_ptr[(5 + k) * grid_len];
          if (prob > maxClassProbs) {
            maxClassId = k;
            maxClassProbs = prob;
          }
        }

        if (maxClassProbs > thres_i8) {
          float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale));
          float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale));
          float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale));
          float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale));
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = exp(box_w) * stride;
          box_h = exp(box_h) * stride;
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                             (deqnt_affine_to_f32(box_confidence, zp, scale)));
          classId.push_back(maxClassId);
          validCount++;
          boxes.push_back(box_x);
          boxes.push_back(box_y);
          boxes.push_back(box_w);
          boxes.push_back(box_h);
        }
      }
    }
  }
  return validCount;
}

int yolox_post_process(int model_width, int model_height, rknn_output *outputs,
                       std::vector<int32_t> &qnt_zps,
                       std::vector<float> &qnt_scales, letterbox_t *letter_box,
                       float conf_threshold, float nms_threshold,
                       object_detect_result_list *od_results) {
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int model_in_w = model_width;
  int model_in_h = model_height;

  memset(od_results, 0, sizeof(object_detect_result_list));

  for (int i = 0; i < 3; i++) {
    grid_h = 80 / pow(2, i);

    grid_w = grid_h;
    stride = model_in_h / grid_h;

    validCount +=
        process_i8((int8_t *)outputs[i].buf, grid_h, grid_w, model_in_h,
                   model_in_w, stride, filterBoxes, objProbs, classId,
                   conf_threshold, qnt_zps[i], qnt_scales[i]);
  }

  // no object detect
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  od_results->count = 0;

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    od_results->results[last_count].box.left =
        (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.top =
        (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].box.right =
        (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.bottom =
        (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;
  return 0;
}

int rtm_post(float *simcc_x_result, float *simcc_y_result,
             cv::Mat affine_transform_reverse,
             std::vector<PosePoint> &pose_result) {

  int extend_width = 384;
  int extend_height = 512;
  if (!pose_result.empty()) {
    pose_result.clear();
  }
  for (int i = 0; i < 17; ++i) {
    // find the maximum and maximum indexes in the value of each Extend_width
    // length
    auto x_biggest_iter =
        std::max_element(simcc_x_result + i * extend_width,
                         simcc_x_result + i * extend_width + extend_width);
    int max_x_pos =
        std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
    int pose_x = max_x_pos / 2;
    float score_x = *x_biggest_iter;

    // find the maximum and maximum indexes in the value of each exten_height
    // length
    auto y_biggest_iter =
        std::max_element(simcc_y_result + i * extend_height,
                         simcc_y_result + i * extend_height + extend_height);
    int max_y_pos =
        std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
    int pose_y = max_y_pos / 2;
    float score_y = *y_biggest_iter;

    // float score = (score_x + score_y) / 2;
    float score = std::max(score_x, score_y);

    PosePoint temp_point;
    temp_point.x = int(pose_x);
    temp_point.y = int(pose_y);
    temp_point.score = score;

    pose_result.emplace_back(temp_point);
  }

  for (int i = 0; i < pose_result.size(); ++i) {
    cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
    origin_point_Mat.at<double>(0, 0) = pose_result[i].x;
    origin_point_Mat.at<double>(1, 0) = pose_result[i].y;

    cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;

    pose_result[i].x = temp_result_mat.at<double>(0, 0);
    pose_result[i].y = temp_result_mat.at<double>(1, 0);
  }

  return 0;
}

int yolox_nano_post_process(int model_width, int model_height,
                            rknn_output *outputs, letterbox_t *letter_box,
                            float conf_threshold, float nms_threshold,
                            std::vector<BOX_RECT> &DetectiontRects) {
  int index = 0;
  int validCount = 0;
  float *data = (float *)outputs[0].buf;
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  if (!DetectiontRects.empty()) {
    DetectiontRects.clear();
  }
  for (int i = 0; i < 3; i++) {
    int grid_h = 52 / pow(2, i); // 52 26 13
    int grid_w = grid_h;
    int stride = model_width / grid_h; // 8 16 32
    for (int grid_y = 0; grid_y < grid_h; grid_y++) {
      for (int grid_x = 0; grid_x < grid_w; grid_x++) {
        float box_confidence = data[index + 4];
        int class_id = 0;
        float max_confidence = 0;
        for (int class_index = 0; class_index < 80; class_index++) {
          float confidence_of_class = data[index + 5 + class_index];
          if (confidence_of_class > max_confidence) {
            max_confidence = confidence_of_class;
            class_id = class_index;
          }
        }
        if (max_confidence >= conf_threshold) {
          float cx = static_cast<float>((data[index + 0] + grid_x) * stride);
          float cy = static_cast<float>((data[index + 1] + grid_y) * stride);
          float w = static_cast<float>(std::exp(data[index + 2]) * stride);
          float h = static_cast<float>(std::exp(data[index + 3]) * stride);
          float x = cx - w / 2;
          float y = cy - h / 2;
          objProbs.push_back(max_confidence * box_confidence);
          classId.push_back(class_id);
          filterBoxes.push_back(x);
          filterBoxes.push_back(y);
          filterBoxes.push_back(w);
          filterBoxes.push_back(h);
          validCount++;
        }
        index += 85;
      }
    }
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];
    if (id == 0 && obj_conf > conf_threshold) {
      BOX_RECT bbox;
      bbox.left = (int)(clamp(x1, 0, model_width) / letter_box->scale);
      bbox.top = (int)(clamp(y1, 0, model_height) / letter_box->scale);
      bbox.right = (int)(clamp(x2, 0, model_width) / letter_box->scale);
      bbox.bottom = (int)(clamp(y2, 0, model_height) / letter_box->scale);
      bbox.prop = obj_conf;
      DetectiontRects.push_back(bbox);
    }
  }
  return 0;
}

int rtm_person_post_process(int model_width, int model_height,
                            rknn_output *outputs, letterbox_t *letter_box,
                            float conf_threshold, float nms_threshold,
                            std::vector<BOX_RECT> &DetectiontRects) {
  int validCount = 0;
  float *data = (float *)outputs[0].buf;
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  if (!DetectiontRects.empty()) {
    DetectiontRects.clear();
  }
  for (int i = 0; i < 3; i++) {
    float *cls_data = (float *)outputs[i].buf;
    float *box_data = (float *)outputs[i + 3].buf;
    int grid_h = 80 / pow(2, i); // 80 40 20
    int grid_w = grid_h;
    int grid_len = grid_h * grid_w;
    int stride = model_width / grid_h; // 8 16 32
    for (int grid_y = 0; grid_y < grid_h; grid_y++) {
      for (int grid_x = 0; grid_x < grid_w; grid_x++) {
        int offset = grid_y * grid_w + grid_x;
        float box_confidence = cls_data[offset];
        box_confidence = sigmoid(box_confidence);
        if (box_confidence > conf_threshold) {
          float *in_ptr = box_data + offset;
          float x1 = grid_x - *in_ptr;
          float y1 = grid_y - in_ptr[grid_len];
          float x2 = grid_x + in_ptr[2 * grid_len];
          float y2 = grid_y + in_ptr[3 * grid_len];
          objProbs.push_back(box_confidence);
          filterBoxes.push_back(x1 / letter_box->scale);
          filterBoxes.push_back(y1 / letter_box->y_scale);
          filterBoxes.push_back(x2 / letter_box->scale);
          filterBoxes.push_back(y2 / letter_box->y_scale);
          classId.push_back(0);
          validCount++;
        }
      }
    }
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1) {
      continue;
    }
    int n = indexArray[i];
    float x1 = filterBoxes[n * 4 + 0];
    float y1 = filterBoxes[n * 4 + 1];
    float x2 = filterBoxes[n * 4 + 2];
    float y2 = filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];
    BOX_RECT bbox;
    bbox.left = (int)(clamp(x1, 0, model_width));
    bbox.top = (int)(clamp(y1, 0, model_height));
    bbox.right = (int)(clamp(x2, 0, model_width));
    bbox.bottom = (int)(clamp(y2, 0, model_height));
    bbox.prop = obj_conf;
    DetectiontRects.push_back(bbox);
  }
  return 0;
}