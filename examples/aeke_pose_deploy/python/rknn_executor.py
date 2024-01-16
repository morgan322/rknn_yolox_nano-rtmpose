from rknn.api import RKNN


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result


    # outputs[0].shape
    # (1, 80, 80, 80)
    # outputs[1].shape
    # (1, 80, 40, 40)
    # outputs[2].shape
    # (1, 80, 20, 20)
    # outputs[3].shape
    # (1, 4, 80, 80)
    # outputs[4].shape
    # (1, 4, 40, 40)
    # (1, 4, 20, 20)
#    index=0, name=output, n_dims=4, dims=[1, 85, 80, 80]
#   index=1, name=788, n_dims=4, dims=[1, 85, 40, 40]
#   index=2, name=output.1, n_dims=4, dims=[1, 85, 20, 20]
    
# logger.info('Start running model on RTMPose...')
#     image_file = "/media/nvidia/D6612D9737620A9A/alg/rtm/python/bus.jpg"
#     device = 'cpu'
#     rknn_path = '/media/nvidia/D6612D9737620A9A/alg/rtm/python/models/rtmdetnp_fp16.rknn'
#     save_path = './media/nvidia/D6612D9737620A9A/alg/rtm/python/busout.jpg'
#     # read image from file
#     logger.info('1. Read image from {}...'.format(image_file))
#     import cv2
#     img = cv2.imread(image_file)

#     dst = np.zeros((640, 640, 3), dtype=np.uint8)
#     resized_img = cv2.resize(
#             img, (640,640), dst, interpolation=cv2.INTER_LINEAR)


#     torch.from_numpy(dst).permute(2, 0, 1).contiguous()



#     # build rknn model
#     logger.info('2. Build rknn model from {}...'.format(rknn_path))
#     from rknn_executor import RKNN_model_container 
#     model = RKNN_model_container(rknn_path,target='rk3588')
    
#     # inference
#     logger.info('4. Inference...')
#     start_time = time.time()
#     outputs = model.run([dst.astype(np.float16)])
#     for i in range(6):
#         print(outputs[i][0][0][0][:10])

#     end_time = time.time()
#     logger.info('4. Inference done, time cost: {:.4f}s'.format(end_time -
#                                                                start_time))
#     for i in range(6):
#             print(outputs[i][0][0][0][:10])
#     outputs = [torch.tensor(element) for element in outputs]
#     cls_score_list = []
#     bbox_pred_list = []
#     score_factor_list = []
#     mlvl_priors = []

#     mlvl_score_factors = None
#     for i in range(3):
#         cls_score_list.append(torch.squeeze(outputs[i], dim=0))
#         bbox_pred_list.append(torch.squeeze(outputs[i + 3]))
#         score_factor_list.append(None)

#         a_list = [torch.arange(0, 640, 8 * 2**i, dtype=torch.float) for _ in range(640 // (8 * 2**i))]
#         b = torch.stack(a_list, dim=1)
#         c = torch.reshape(b, (b.shape[0] * b.shape[1], 1))

#         d_list=[i for i in range(0,640,8 * 2**i)] * (640 //  (8 * 2**i))
#         tensor = torch.tensor(d_list, dtype=torch.float)
#         column_vector = torch.reshape(tensor, (len(d_list), 1))
#         e = torch.cat([column_vector, c], dim=1)
#         mlvl_priors.append(e)
    
#     mlvl_bbox_preds = []
#     mlvl_valid_priors = []
#     mlvl_scores = []
#     mlvl_labels = []
#     print(cls_score_list[0].size(),bbox_pred_list[0].size())
#     for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
#                 enumerate(zip(cls_score_list, bbox_pred_list,
#                             score_factor_list, mlvl_priors)):

#         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

#         dim = 4
#         bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
#         cls_out_channels = 1
#         print(cls_score.size())

#         cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_out_channels)

#         scores = cls_score.sigmoid()
        
#         nms_pre = 30000
#         score_thr = 0.001

#         results = filter_scores_and_topk(
#             scores, score_thr, nms_pre,
#             dict(bbox_pred=bbox_pred, priors=priors))
#         scores, labels, keep_idxs, filtered_results = results

#         bbox_pred = filtered_results['bbox_pred']
#         priors = filtered_results['priors']

#         mlvl_bbox_preds.append(bbox_pred)
#         mlvl_valid_priors.append(priors)
#         mlvl_scores.append(scores)
#         mlvl_labels.append(labels)

#     bbox_pred = torch.cat(mlvl_bbox_preds)
#     priors = torch.cat(mlvl_valid_priors, dim=0)

#     distance = bbox_pred
#     points = priors
#     x1 = points[..., 0] - distance[..., 0]
#     y1 = points[..., 1] - distance[..., 1]
#     x2 = points[..., 0] + distance[..., 2]
#     y2 = points[..., 1] + distance[..., 3]

#     bboxes = torch.stack([x1, y1, x2, y2], -1)

#     max_shape = (640,640)
#     if max_shape is not None:
#         if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
#             # speed up
#             bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
#             bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
#     scale_factor = [1.265625, 1.6875]
#     repeat_num = int(bboxes.size(-1) / 2)
#     scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))

#     bboxes = bboxes * scale_factor
#     scores = torch.cat(mlvl_scores)
#     labels = torch.cat(mlvl_labels)

#     # from mmcv.ops import batched_nms
#     # boxes,keep = batched_nms(bboxes, scores, labels, {'type': 'nms', 'iou_threshold': 0.65})
#     # results.scores = boxes[:, -1]


#     import torchvision.ops as ops
#     idx = ops.batched_nms(bboxes, scores, labels, 0.65)


#     bboxes = bboxes[idx]
#     scores = scores[idx]
#     labels = labels[idx]
#     mask = scores >= 0.4
#     bboxes = bboxes[mask]
#     scores = scores[mask]
#     labels = labels[mask] 