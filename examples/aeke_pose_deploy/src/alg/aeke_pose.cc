#include "postprocess.h"
#include "preprocess.h"
#include "rknn_api.h"
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <sys/time.h>
#include <android/log.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "aeke_pose.hpp"
#include "coreNum.hpp"

static unsigned char *model_data;

static void dump_tensor_attr(rknn_tensor_attr *attr) {
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  LOGI("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, "
         "w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(),
         attr->n_elems, attr->size, attr->w_stride, attr->size_with_stride,
         get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size) {
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size) {
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

aekePose::aekePose(const std::string &det_model_path,
                   const std::string &pose_model_path) {
  this->det_model_path = det_model_path;
  this->pose_model_path = pose_model_path;
  nms_threshold = NMS_THRESH;      // 默认的NMS阈值
  box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
}

static int init_model(const std::string &model_path,
                      rknn_app_context_t *app_ctx) {
  int ret;
  rknn_context ctx = 0;

  int model_data_size = 0;
  model_data = load_model(model_path.c_str(), &model_data_size);
  ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_core_mask core_mask;
  switch (get_core_num()) {
  case 0:
    core_mask = RKNN_NPU_CORE_0;
    break;
  case 1:
    core_mask = RKNN_NPU_CORE_1;
    break;
  case 2:
    core_mask = RKNN_NPU_CORE_2;
    break;
  }
  ret = rknn_set_core_mask(ctx, core_mask);
  if (ret < 0) {
    printf("rknn_init core error ret=%d\n", ret);
    return -1;
  }

  // Get Model Input Output Number
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  // Get Model Input Info
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  // Get Model Output Info

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(output_attrs[i]));
  }

  // Set to context
  app_ctx->rknn_ctx = ctx;

  app_ctx->io_num = io_num;
  app_ctx->input_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->input_attrs, input_attrs,
         io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx->output_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->output_attrs, output_attrs,
         io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    app_ctx->model_channel = input_attrs[0].dims[1];
    app_ctx->model_height = input_attrs[0].dims[2];
    app_ctx->model_width = input_attrs[0].dims[3];
  } else {
    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];
  }

  return 0;
}

int aekePose::init() {

  int ret = init_model(det_model_path, &det_ctx);
  ret = init_model(pose_model_path, &pose_ctx);

  return 0;
}

static int det_infer(rknn_app_context_t *app_ctx, float nms_threshold,
                     float box_conf_threshold, cv::Mat image, FrameInfo &Info,
                     letterbox_t &letter_box) {
  rknn_input inputs[app_ctx->io_num.n_input];
  rknn_output outputs[app_ctx->io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size =
      app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
  inputs[0].buf = image.data;
  rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);

  for (int i = 0; i < app_ctx->io_num.n_output; i++) {
    outputs[i].want_float = 1;
  }
  int ret = rknn_run(app_ctx->rknn_ctx, NULL);
  ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs,
                         NULL);
  ret = yolox_nano_post_process(app_ctx->model_width, app_ctx->model_height,
                                outputs, &letter_box, box_conf_threshold,
                                nms_threshold, Info.DetectiontRects);
  rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
  return 0;
}

static int pose_infer(rknn_app_context_t *app_ctx, cv::Mat image,
                      cv::Mat affine_transform_reverse, FrameInfo &Info) {
  rknn_input inputs[app_ctx->io_num.n_input];
  rknn_output outputs[app_ctx->io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size =
      app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
  inputs[0].buf = image.data;
  rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);

  for (int i = 0; i < app_ctx->io_num.n_output; i++) {
    outputs[i].want_float = 1;
  }
  int ret = rknn_run(app_ctx->rknn_ctx, NULL);
  ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs,
                         NULL);
  float *simcc_x_result = (float *)outputs[0].buf;
  float *simcc_y_result = (float *)outputs[1].buf;
  ret = rtm_post(simcc_x_result, simcc_y_result, affine_transform_reverse,
                 Info.pose_result);
  rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
  return 0;
}

static int rtm_infer(rknn_app_context_t *app_ctx, float nms_threshold,
                     float box_conf_threshold, cv::Mat image, FrameInfo &Info,
                     letterbox_t &letter_box) {
  rknn_input inputs[app_ctx->io_num.n_input];
  rknn_output outputs[app_ctx->io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size =
      app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
  inputs[0].buf = image.data;
  rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);

  for (int i = 0; i < app_ctx->io_num.n_output; i++) {
    outputs[i].want_float = 1;
  }
  int ret = rknn_run(app_ctx->rknn_ctx, NULL);
  ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs,
                         NULL);
  ret = rtm_person_post_process(app_ctx->model_width, app_ctx->model_height,
                                outputs, &letter_box, box_conf_threshold,
                                nms_threshold, Info.DetectiontRects);
  rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
  return 0;
}

// cv::Mat aekePose::infer(cv::Mat &orig_img)
FrameInfo aekePose::infer(FrameInfo &Info) {
  std::lock_guard<std::mutex> lock(mtx);
  cv::Mat yuvMat = Info.orig_img;
  cv::Mat orig_img;
  cv::cvtColor(yuvMat, orig_img, cv::COLOR_YUV2RGB_NV21);
  cv::rotate(orig_img, orig_img, cv::ROTATE_90_CLOCKWISE);
  img_width = orig_img.cols;
  img_height = orig_img.rows;
  int is_quan;
  letterbox_t letter_box;
  // object_detect_result_list od_results;

  // cv::Mat orig_img;
  // rgbMat.copyTo(orig_img);
  if (Info.alg_type == AlgoType::kRtmpose) {
    for (int i = 0; i < Info.DetectiontRects.size(); i++) {
      is_quan = 1;
      std::pair<cv::Mat, cv::Mat> crop_result_pair =
          CropImageByDetectBox(orig_img, Info.DetectiontRects[i]);
      cv::Mat crop_mat = crop_result_pair.first;
      cv::Mat affine_transform_reverse = crop_result_pair.second;

      cv::Mat input_mat_copy_rgb;
      cv::cvtColor(crop_mat, input_mat_copy_rgb, cv::COLOR_BGR2RGB);
      ret = pose_infer(&pose_ctx, input_mat_copy_rgb, affine_transform_reverse,
                       Info);
    }
  } else if (Info.alg_type ==  AlgoType::kYolox) {
    
    is_quan = 1;
    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Size target_size(det_ctx.model_width, det_ctx.model_height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / orig_img.cols;
    float scale_h = (float)target_size.height / orig_img.rows;
    // cv::cvtColor(img, orig_img, cv::COLOR_BGR2RGB);
    // 图像缩放/Image scaling
    
    if (img_width != det_ctx.model_width ||
        img_height != det_ctx.model_height) {
      // // rga
      rga_buffer_t src;
      rga_buffer_t dst;
      memset(&src, 0, sizeof(src));
      memset(&dst, 0, sizeof(dst));
      ret = resize_rga(src, dst, orig_img, resized_img, target_size);
      if (ret != 0) {
        fprintf(stderr, "resize with rga error\n");
      }

      float min_scale = std::min(scale_w, scale_h);
      // scale_w = min_scale;
      // scale_h = min_scale;
      letter_box.x_pad = (int)(det_ctx.model_width - min_scale * img_width)/2 ;
      letter_box.y_pad = (int)(det_ctx.model_height - min_scale * img_height)/2 ;
      letter_box.scale = min_scale;
      // opencv
      // letterbox(orig_img, resized_img, pads, min_scale, target_size);

      // inputs[0].buf = resized_img.data;
      // cv::Mat image(height, width, CV_8UC3, resized_img.data);
      // cv::imwrite("opencv.jpg", image);

      ret = det_infer(&det_ctx, nms_threshold, box_conf_threshold, resized_img,
                      Info, letter_box);
    }
  } 

  return Info;
}

aekePose::~aekePose() {
  deinitPostProcess();

  if (det_ctx.rknn_ctx != 0) {
    rknn_destroy(det_ctx.rknn_ctx);
    det_ctx.rknn_ctx = 0;
  }
  if (det_ctx.input_attrs != NULL) {
    free(det_ctx.input_attrs);
    det_ctx.input_attrs = NULL;
  }
  if (det_ctx.output_attrs != NULL) {
    free(det_ctx.output_attrs);
    det_ctx.output_attrs = NULL;
  }

  if (pose_ctx.rknn_ctx != 0) {
    rknn_destroy(pose_ctx.rknn_ctx);
    pose_ctx.rknn_ctx = 0;
  }
  if (pose_ctx.input_attrs != NULL) {
    free(pose_ctx.input_attrs);
    pose_ctx.input_attrs = NULL;
  }
  if (pose_ctx.output_attrs != NULL) {
    free(pose_ctx.output_attrs);
    pose_ctx.output_attrs = NULL;
  }

  if (model_data)
    free(model_data);
}
