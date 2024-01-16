import argparse
import time
from typing import List, Tuple
import torch
import cv2
import loguru
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN
from coco_utils import COCO_test_helper
import torchvision.ops as ops

logger = loguru.logger

OBJ_THRESH = 0.022
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]



def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)

    return sess

def inference(sess: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    """Inference RTMPose model.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    # build input
    input = [img.transpose(2, 0, 1)]

    # build output
    sess_input = {sess.get_inputs()[0].name: input}
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # run model
    outputs = sess.run(sess_output, sess_input)

    return outputs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    # candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    # classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score*box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = box_class_probs[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    box_xy = position[:,:2,:,:]
    box_wh = np.exp(position[:,2:4,:,:]) * stride

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy

# def post_process(input_data):
#     boxes, scores, classes_conf = [], [], []

#     input_data = [_in.reshape([1, -1]+list(_in.shape[-2:])) for _in in input_data]
#     for i in range(len(input_data)):
#         boxes.append(box_process(input_data[i][:,:4,:,:]))
#         scores.append(input_data[i][:,4:5,:,:])
#         classes_conf.append(input_data[i][:,5:,:,:])

#     def sp_flatten(_in):
#         ch = _in.shape[1]
#         _in = _in.transpose(0,2,3,1)
#         return _in.reshape(-1, ch)

#     boxes = [sp_flatten(_v) for _v in boxes]
#     classes_conf = [sp_flatten(_v) for _v in classes_conf]
#     scores = [sp_flatten(_v) for _v in scores]

#     boxes = np.concatenate(boxes)
#     classes_conf = np.concatenate(classes_conf)
#     scores = np.concatenate(scores)

#     # filter according to threshold
#     boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

#     # nms
#     nboxes, nclasses, nscores = [], [], []
#     keep = nms_boxes(boxes, scores)
#     if len(keep) != 0:
#         nboxes.append(boxes[keep])
#         nclasses.append(classes[keep])
#         nscores.append(scores[keep])

#     if not nclasses and not nscores:
#         return None, None, None

#     boxes = np.concatenate(nboxes)
#     classes = np.concatenate(nclasses)
#     scores = np.concatenate(nscores)

#     return boxes, classes, scores

def nms(boxes, scores, threshold):
    sorted_idxs = np.argsort(scores)[::-1]
    keep_idxs = []

    while len(sorted_idxs) > 0:
        current_idx = sorted_idxs[0]
        keep_idxs.append(current_idx)

        current_box = boxes[current_idx]
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])

        sorted_idxs = sorted_idxs[1:]
        overlapping_idxs = []

        for idx in sorted_idxs:
            box = boxes[idx]
            area = (box[2] - box[0]) * (box[3] - box[1])

            x0 = max(current_box[0], box[0])
            y0 = max(current_box[1], box[1])
            x1 = min(current_box[2], box[2])
            y1 = min(current_box[3], box[3])

            overlap_area = max(x1 - x0, 0) * max(y1 - y0, 0)
            overlap_ratio = overlap_area / (current_area + area - overlap_area)

            if overlap_ratio > threshold:
                overlapping_idxs.append(idx)

        sorted_idxs = np.delete(sorted_idxs, overlapping_idxs)

    return keep_idxs




def bbox_transform_inv(bboxes):
    bboxes = bboxes.copy()
    w = bboxes[:, 2] - bboxes[:, 0] + 1e-6
    h = bboxes[:, 3] - bboxes[:, 1] + 1e-6
    bboxes[:, 0] -= 0.5 * w
    bboxes[:, 1] -= 0.5 * h
    bboxes[:, 2] += 0.5 * w
    bboxes[:, 3] += 0.5 * h
    return bboxes

def clip_boxes(bboxes, img_shape=None):
    if img_shape is None:
        return bboxes.clip(min=0)
    else:
        h, w = img_shape[:2]
        bboxes[:, 0] = np.clip(bboxes[:, 0], a_min=0, a_max=w-1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], a_min=0, a_max=h-1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], a_min=0, a_max=w-1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], a_min=0, a_max=h-1)
        return bboxes

def non_max_suppression(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])
        idx = np.where(ious <= iou_threshold)[0]
        order = order[idx + 1]
    return np.array(keep, dtype=np.int32)

def compute_iou(boxes1, boxes2):
    lt = np.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = np.maximum(rb - lt + 1e-6, 0)
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1e-6) * (boxes1[:, 3] - boxes1[:, 1] + 1e-6)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1e-6) * (boxes2[:, 3] - boxes2[:, 1] + 1e-6)
    iou = overlap / (area1 + area2 - overlap)
    return iou


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        if score > OBJ_THRESH:
            top, left, right, bottom = [int(_b) for _b in box]
            # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite('rtmdet.jpg', image)

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results

def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def onnx_main():
    

    logger.info('Start running model on RTMPose...')
    image_file = "/media/nvidia/D6612D9737620A9A/program/aeke_pose/examples/aeke_pose_deploy/data/bus.jpg"
    device = 'cpu'
    onnx_file = '/media/nvidia/D6612D9737620A9A/program/aeke_pose/examples/aeke_pose_deploy/model/det/rtmdetnp.onnx'
    save_path = './media/nvidia/D6612D9737620A9A/alg/rtm/python/busout.jpg'
    # read image from file
    logger.info('1. Read image from {}...'.format(image_file))
    img = cv2.imread(image_file)

    dst = np.zeros((640, 640, 3), dtype=np.uint8)
    resized_img = cv2.resize(
            img, (640,640), dst, interpolation=cv2.INTER_LINEAR)


    torch.from_numpy(dst).permute(2, 0, 1).contiguous()

    # build onnx model
    logger.info('2. Build onnx model from {}...'.format(onnx_file))
    sess = build_session(onnx_file, device)
    h, w = sess.get_inputs()[0].shape[2:]


    # preprocessing
    logger.info('3. Preprocess image...')
    co_helper = COCO_test_helper(enable_letter_box=True)
    # preprocessing
    logger.info('3. Preprocess image...')
    pad_color = (0,0,0)
    # img = co_helper.letter_box(im= img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    im= img.copy()
    # cv2.imwrite("letter_box.jpg",img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # inference
    logger.info('4. Inference...')
    start_time = time.time()
    outputs = inference(sess, dst)

    for i in range(6):
        print(outputs[i][0][0][0][:10])
    outputs = [torch.tensor(element) for element in outputs]
    cls_score_list = []
    bbox_pred_list = []
    score_factor_list = []
    mlvl_priors = []

    mlvl_score_factors = None
    for i in range(3):
        cls_score_list.append(torch.squeeze(outputs[i]))
        bbox_pred_list.append(torch.squeeze(outputs[i + 3]))
        score_factor_list.append(None)

        a_list = [torch.arange(0, 640, 8 * 2**i, dtype=torch.float) for _ in range(640 // (8 * 2**i))]
        b = torch.stack(a_list, dim=1)
        c = torch.reshape(b, (b.shape[0] * b.shape[1], 1))

        d_list=[i for i in range(0,640,8 * 2**i)] * (640 //  (8 * 2**i))
        tensor = torch.tensor(d_list, dtype=torch.float)
        column_vector = torch.reshape(tensor, (len(d_list), 1))
        e = torch.cat([column_vector, c], dim=1)
        mlvl_priors.append(e)
    
    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    mlvl_labels = []
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        dim = 4
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
        cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, 1)

        scores = cls_score.sigmoid()
        
        nms_pre = 30000
        score_thr = 0.001

        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = torch.cat(mlvl_valid_priors, dim=0)

    distance = bbox_pred
    points = priors
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)
    max_shape = IMG_SIZE
    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])

    # scale_factor = [1.265625, 1.6875]
    # repeat_num = int(bboxes.size(-1) / 2)
    # scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))

    bboxes = bboxes 
    scores = torch.cat(mlvl_scores)
    labels = torch.cat(mlvl_labels)

    # np.savetxt("bboxes.txt", bboxes.numpy(), delimiter=",")
    # np.savetxt("scores.txt", scores.numpy(), delimiter=",")
    # np.savetxt("labels.txt", labels.numpy(), delimiter=",")


    # boxes, classes, scores = filter_boxes(bboxes.numpy(), scores.numpy(), labels.numpy())

    # nboxes, nclasses, nscores = [], [], []
    # keep = nms_boxes(boxes, classes)
    # if len(keep) != 0:
    #     nboxes.append(bboxes[keep])
    #     nclasses.append(labels.numpy()[keep])
    #     nscores.append(scores[keep])
    # boxes = np.concatenate(nboxes)
    # classes = np.concatenate(nclasses)
    # scores = np.concatenate(nscores)

    # draw(im, bboxes.numpy(), scores.numpy(), labels.numpy())

    end_time = time.time()
    logger.info('4. Inference done, time cost: {:.4f}s'.format(end_time -
                                                               start_time))


    logger.info('Done...')

def rknn_main():
    

    logger.info('Start running model on RTMPose...')
    image_file = "/media/nvidia/D6612D9737620A9A/program/aeke_pose/examples/aeke_pose_deploy/data/bus.jpg"
    rknn_path = '/media/nvidia/D6612D9737620A9A/program/aeke_pose/examples/aeke_pose_deploy/model/det/rtmdetnp_fp16.rknn'

    # read image from file
    logger.info('1. Read image from {}...'.format(image_file))
    import cv2
    import torch
    img = cv2.imread(image_file)

    dst = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.resize(img, (640, 640), dst, interpolation=cv2.INTER_LINEAR)

    np.transpose(dst, (2, 0, 1))
    
    # build rknn model
    logger.info('2. Build rknn model from {}...'.format(rknn_path))
    from rknn_executor import RKNN_model_container
    model = RKNN_model_container(rknn_path,target='rk3588')

    # inference
    logger.info('4. Inference...')
    start_time = time.time()
    outputs = model.run([dst.astype(np.float16)])
    end_time = time.time()
    logger.info('4. Inference done, time cost: {:.4f}s'.format(end_time - start_time))

    outputs = [np.array(element) for element in outputs]
    cls_score_list = []
    bbox_pred_list = []

    mlvl_priors = []

    for i in range(3):
        cls_score_list.append(np.squeeze(outputs[i], axis=0))
        bbox_pred_list.append(np.squeeze(outputs[i + 3]))

        a_list = [np.arange(0, 640, 8 * 2**i, dtype=np.float) for _ in range(640 // (8 * 2**i))]
        b = np.stack(a_list, axis=1)
        c = np.reshape(b, (b.shape[0] * b.shape[1], 1))

        d_list=[i for i in range(0,640,8 * 2**i)] * (640 //  (8 * 2**i))
        tensor = np.array(d_list, dtype=np.float)
        column_vector = np.reshape(tensor, (len(d_list), 1))
        e = np.concatenate([column_vector, c], axis=1)
        mlvl_priors.append(e)

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    mlvl_labels = []
    cls_out_channels = 1
    dim = 4
    for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                            mlvl_priors)):

        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

        
        bbox_pred = np.transpose(bbox_pred, (1, 2, 0)).reshape(-1, dim)
        
        
        cls_score = np.transpose(cls_score, (1, 2, 0)).reshape(-1, cls_out_channels)

        scores = 1 / (1 + np.exp(-cls_score))
        
        nms_pre = 30000
        score_thr = 0.001

        results = filter_scores_and_topk(
            torch.tensor(scores), score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)

    bbox_pred = np.concatenate(mlvl_bbox_preds)
    priors = np.concatenate(mlvl_valid_priors, axis=0)

    distance = bbox_pred
    points = priors
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = np.stack([x1, y1, x2, y2], -1)

    max_shape = (640, 640)
    if max_shape is not None:
        if bboxes.ndim == 2:
            # speed up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], a_min=0, a_max=max_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], a_min=0, a_max=max_shape[0])
    scale_factor = [1.265625, 1.6875]
    repeat_num = int(bboxes.shape[-1] / 2)
    scale_factor = np.tile(scale_factor, (1, repeat_num))

    bboxes = bboxes * scale_factor
    scores = np.concatenate(mlvl_scores)
    labels = np.concatenate(mlvl_labels)

    # 将bboxes、scores和labels转换为PyTorch张量
    bboxes = torch.from_numpy(bboxes)
    scores = torch.from_numpy(scores).to(bboxes.dtype)  # 将scores转换为与bboxes相同的数据类型
    labels = torch.from_numpy(labels)

    # 使用batched_nms函数
    idx = ops.batched_nms(bboxes, scores, labels, 0.65)


    bboxes = bboxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    mask = scores >= 0.4
    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
        "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
        "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
        "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
        "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
        "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
        "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

    import cv2

    image = cv2.imread(image_file)
    for box, score, cl in zip(bboxes.numpy(), scores.numpy(), labels.numpy()):
    
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite('rtmdet.jpg', image)

    

if __name__ == '__main__':

    # onnx_main()
    rknn_main()
