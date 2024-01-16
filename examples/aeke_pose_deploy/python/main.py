from yolox_nano import YoloxONNX
import cv2
from bytetrack-opencv-onnxruntime.onnxruntime.python.byte_tracker.tracker.byte_tracker import BYTETracker


image_file = "/media/nvidia/D6612D9737620A9A/program/rknpu2/examples/aeke_pose_deploy/resize_nano.jpg"

onnx_file = '/media/nvidia/D6612D9737620A9A/alg/rtm/python/models/yolox_nano.onnx'

yolox = YoloxONNX(
        model_path=onnx_file,
    )
frame = cv2.imread(image_file)
boxes, confidences, class_ids = yolox.inference(frame)

# 选择主目标
main_object_idx = confidences.index(max(confidences))
main_object_box = boxes[main_object_idx]

tracker = cv2.TrackerKCF_create()
# 初始化目标跟踪器，并指定主目标的初始位置
tracker.init(frame, tuple(main_object_box))

# 处理后续帧
while True:
    ret, frame = video.read()
    if not ret:
        break
        
    # 目标跟踪
    success, box = tracker.update(frame)
    
    if success:
        # 更新主目标位置
        main_object_box = box

        # 在图像中绘制跟踪框
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Main Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示结果
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()