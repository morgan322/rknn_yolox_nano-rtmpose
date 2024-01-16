#include <iostream>
#include <memory>
#include <stdio.h>
#include <sys/time.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

// #include "opencv2/core/core.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

#include "BYTETracker.h"
#include "aeke_pose.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "preprocess.h"
#include "rknnPool.hpp"

int main(int argc, char **argv) {

  char *det_model_name = (char *)argv[1];
  char *pose_model_name = (char *)argv[2];
  char *vedio_name = argv[3];
  int threadNum = std::atoi(argv[4]);

  rknnPool<aekePose, FrameInfo, FrameInfo> RknnPool_(
      det_model_name, pose_model_name, threadNum);

  if (RknnPool_.init() != 0) {
    printf("rknnPool init fail!\n");
    return 1;
  }

  int fps = 30;
  BYTETracker tracker(fps, 30);

  cv::Mat src_img = cv::imread(vedio_name, 1);

  FrameInfo frame;
  frame.width = src_img.cols; // 列
  frame.height = src_img.rows;
  frame.orig_img = src_img;
  frame.in_type = InputType::kImagergb;

  // frame.width = 480;
  // frame.height = 640;
  // std::vector<char> buffer = getyuv((std::string)vedio_name);
  // frame.orig_img = convertYUVtoJpg(static_cast<void *>(buffer.data()),
  //                                  frame.width, frame.height, buffer.size());
  // cv::imwrite("nature.jpg",frame.orig_img);                           
  // int center_x = frame.width / 2;
  // int center_y = frame.height / 2;
  // cv::Mat rotation_matrix =
  //     cv::getRotationMatrix2D(cv::Point(center_x, center_y), -90, 1.0);
  // cv::warpAffine(frame.orig_img, frame.orig_img, rotation_matrix,
  //                cv::Size(frame.height, frame.width));

  // cv::imwrite("warpAffine.jpg",frame.orig_img);

  int frames = 0;
  struct timeval time;
  gettimeofday(&time, nullptr);
  auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
  auto beforeTime = startTime;

  while (true) {

    frame.alg_type = AlgoType::kYolox;
    // frame.alg_type = AlgoType::kRtmdet;
    if (RknnPool_.put(frame) != 0)
      break;
    if (frames >= threadNum && RknnPool_.get(frame) != 0)
      break;
    frames++;
    
    vector<Object> objects;
    for (int i = 0; i < frame.DetectiontRects.size(); i++) {
      Object obj;
      obj.rect.x = frame.DetectiontRects[i].left;
      obj.rect.y = frame.DetectiontRects[i].top;
      obj.rect.width =
          frame.DetectiontRects[i].right - frame.DetectiontRects[i].left;
      obj.rect.height =
          frame.DetectiontRects[i].bottom - frame.DetectiontRects[i].top;
      obj.prob = frame.DetectiontRects[i].prop;
      // printf("(%d,%d,%d,%d)\t",frame.DetectiontRects[i].left, frame.DetectiontRects[i].top,frame.DetectiontRects[i].right, frame.DetectiontRects[i].bottom);
      // cv::rectangle(
			// 	src_img,
			// 	cv::Point(frame.DetectiontRects[i].left, frame.DetectiontRects[i].top),
			// 	cv::Point(frame.DetectiontRects[i].right, frame.DetectiontRects[i].bottom),
			// 	cv::Scalar{ 255, 0, 0 },
			// 	2);
      obj.label = 0;
      objects.push_back(obj);
    }
    vector<STrack> output_stracks = tracker.update(objects);
    for (int i = 0; i < output_stracks.size(); i++) {
      vector<float> tlwh = output_stracks[i].tlwh;
      bool vertical = tlwh[2] / tlwh[3] > 1.6;
      // if (tlwh[2] * tlwh[3] > 20 && !vertical) {
      //   Scalar s = tracker.get_color(output_stracks[i].track_id);
      //   putText(src_img, format("%d", output_stracks[i].track_id),
      //           Point(tlwh[0], tlwh[1] - 5), 0, 0.6, Scalar(0, 0, 255), 2,
      //           LINE_AA);
      //   rectangle(src_img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
      // }
    }

    frame.alg_type = AlgoType::kRtmpose;
    if (RknnPool_.put(frame) != 0)
      break;
    if (frames >= threadNum && RknnPool_.get(frame) != 0)
      break;
    frames++;
    printf("%d,%d,%d\t",frame.DetectiontRects.size(),frame.pose_result.size(),output_stracks.size());
    // if (frames > 1) {break;}
    // cv::imwrite("obj.jpg",src_img);
    if (frames % 120 == 0) {
      gettimeofday(&time, nullptr);
      auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
      printf("平均帧率:\t %f fps/s\n",
             120.0 / float(currentTime - beforeTime) * 1000.0 * 0.5);
      beforeTime = currentTime;
      // break;
    }
  }

  // Clear the thread pool
  while (true) {
    FrameInfo frame;
    if (RknnPool_.get(frame) != 0)
      break;
  }

  return 0;
}