#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"
#include <iostream>
#include <jni.h>
#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

struct PosePoint {
  int x;
  int y;
  float score;

  PosePoint() {
    x = 0;
    y = 0;
    score = 0.0;
  }
};

typedef struct {
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;
typedef struct _BOX_RECT {
  int left;
  int right;
  int top;
  int bottom;
  float prop;
} BOX_RECT;

typedef struct {
  BOX_RECT box;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct {
  int id;
  int count;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
  int x_pad;
  int y_pad;
  float scale;
  float y_scale;
  int img_w;
  int img_h;
} letterbox_t;

typedef struct __detect_result_t {
  char name[OBJ_NAME_MAX_SIZE];
  BOX_RECT box;
  float prop;
} detect_result_t;

typedef struct _detect_result_group_t {
  int id;
  int count;
  detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h,
                 int model_in_w, float conf_threshold, float nms_threshold,
                 BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

void deinitPostProcess();

// 输入帧类型枚举
enum class InputType { kImagergb = 0, kImageyuv, kVideo, kAudio, kUndefined };

// 算法类型枚举
enum class AlgoType { kYolov8pose = 0, kRtmpose, kYolox, kRtmdet, kUndefined };

struct AlgParms {
  char *mod_path_det; // 模型文件路径
  char *mod_path_pose;
  float mod_thres{0};
  float frame_rate{15};
  int batch_size{1};
  int thread_num{1};
};

struct FrameInfo {
  int id;
  // void *yuvdata;
  jbyte *yuvdata;
  size_t size;
  cv::Mat orig_img;
  int width;
  int height;
  std::vector<BOX_RECT> DetectiontRects;
  std::vector<PosePoint> pose_result;
  InputType in_type;
  AlgoType alg_type;
  int traker_id;
};

// yolov8pose
typedef struct {
  float x;
  float y;
  float score;
} yoloKeyPoint;

typedef struct {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float score;
  int classId;
  std::vector<yoloKeyPoint> keyPoints;
} DetectRect;

class GetResultRectYolov8 {
public:
  GetResultRectYolov8();

  ~GetResultRectYolov8();

  int GenerateMeshgrid();

  int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp,
                             std::vector<float> &qnt_scale,
                             std::vector<float> &DetectiontRects,
                             std::vector<float> &DetectKeyPoints, int img_width,
                             int img_height);

  float sigmoid(float x);

private:
  std::vector<float> meshgrid;

  const int class_num = 1;
  int headNum = 3;

  int input_w = 640;
  int input_h = 640;
  int strides[3] = {8, 16, 32};
  int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
  int keypoint_num = 17;

  float nmsThresh = 0.45;
  float objectThresh = 0.1;
};
int rtm_post(float *simcc_x_result, float *simcc_y_result,
             cv::Mat affine_transform_reverse,
             std::vector<PosePoint> &pose_result);
int yolox_post_process(int model_width, int model_height, rknn_output *outputs,
                       std::vector<int32_t> &qnt_zps,
                       std::vector<float> &qnt_scales, letterbox_t *letter_box,
                       float conf_threshold, float nms_threshold,
                       object_detect_result_list *od_results);

int yolox_nano_post_process(int model_width, int model_height,
                            rknn_output *outputs, letterbox_t *letter_box,
                            float conf_threshold, float nms_threshold,
                            std::vector<BOX_RECT> &DetectiontRects);
int rtm_person_post_process(int model_width, int model_height,
                            rknn_output *outputs, letterbox_t *letter_box,
                            float conf_threshold, float nms_threshold,
                            std::vector<BOX_RECT> &DetectiontRects);
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
