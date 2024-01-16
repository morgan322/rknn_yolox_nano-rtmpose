#ifndef _RKNN_YOLOV5_DEMO_PREPROCESS_H_
#define _RKNN_YOLOV5_DEMO_PREPROCESS_H_

#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include <stdio.h>
#include <android/log.h>

typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;
typedef struct {
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
} image_buffer_t;

#define TAG "AEKE_AI_LOG"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)


void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads,
               const float scale, const cv::Size &target_size,
               const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image,
               cv::Mat &resized_image, const cv::Size &target_size);

cv::Mat convertYUVtoJpg(const void *yuvData, int width, int height,
                        size_t size);

cv::Mat img_cut(cv::Mat &image, int x1, int y1, int x2, int y2);

std::vector<char> getyuv(const std::string &filePath);
cv::Mat convert(jbyte *yuvData);
cv::Mat rotate(cv::Mat rgbMat);
int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color);
int get_image_size(image_buffer_t *image);
std::pair<cv::Mat, cv::Mat>
CropImageByDetectBox(const cv::Mat &input_image, const BOX_RECT &box);
cv::Mat GetAffineTransform(float center_x, float center_y,
                                  float scale_width, float scale_height,
                                  int output_image_width,
                                  int output_image_height, bool inverse) ;
#endif //_RKNN_YOLOV5_DEMO_PREPROCESS_H_
