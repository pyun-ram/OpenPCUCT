import numpy as np
from typing import *
from itertools import chain
from copy import deepcopy

from pcdet.datasets.kitti.kitti_object_eval_python.eval import \
    rotate_iou_gpu_eval, d3_box_overlap, image_box_overlap, \
    bev_box_overlap, get_split_parts, print_str, \
    fused_compute_statistics, compute_statistics_jit, \
    get_mAP_R40, get_mAP, get_thresholds, _prepare_data

from pcuct.utils.debug_utils import Timer, Debug
from pcuct.ops.jiou.jiou_utils import jiou_eval_bev, jiou_eval_3d

def bev_box_overlap_uct(
    boxes:np.ndarray, qboxes:np.ndarray,
    ucts:Dict[str,np.ndarray], qucts:Dict[str,np.ndarray],
    calibs:List, qcalibs:List,
    dcmasks:List[bool], qdcmasks:List[bool],
    frameids:List[str], qframeids:List[str],
    criterion:int=-1,
    bool_delta:bool=False, is_boxes_gt:bool=False) -> np.ndarray:
    '''
    Evaluate overlap between boxes and qboxes considering the uncertainties.
    Args:
        boxes (N, 5): BEV bounding boxes (in left Camera frame) [xc, yc, l, w, ry]
        qboxes (M, 5): BEV bounding boxes (in left Camera frame) [xc, yc, l, w, ry] (query)
        ucts (N): predictive distributions (in LiDAR frame). Each item contains a dict:
            {"mean": np.ndarray (1,6)/(1,5),
             "cov": np.ndarray (6,6)/(5,5)}
        qucts (M): predictive distributions (in LiDAR frame) (query). Each item contains a dict:
            {"mean": np.ndarray (1,6)/(1,5),
             "cov": np.ndarray (6,6)/(5,5)}
        calibs (List[KittiCalib])(N): calib classes
        qcalibs (List[KittiCalib])(M): calib classes (query)
        dcmasks (N): it indicates whether it is a 'DontCare' class.
        qdcmasks (M): it indicates whether it is a 'DontCare' class. (query)
        frameids (N): frame ids
        qfameids (M): frame ids (query)
        criterion:
        bool_delta (bool): if True, make gts as a delta distribution
        is_boxes_gt (bool): True indicates <boxes> are labels. Otherwise
         it indicates <qboxes> are labels. It will take effect in computing delta JIoU.
    Returns:
        riou (N, M): IoU matrix
    TODO: This function can be further improved:
        - if it computes JIoU in the camera frame;
    '''
    def _transform(_boxes, _calibs):
        '''
        loc, dims, rots
        x z  l w
        '''
        _boxes_Flidar = []
        for box, calib in zip(_boxes, _calibs):
            centers_Fcam = np.array([[box[0], 0, box[1]]])
            centers_Flidar = calib.leftcam2lidar(centers_Fcam)
            tmp = deepcopy(box)
            tmp[:2] = centers_Flidar[0, :2]
            tmp[-1] = np.pi /2.0 - tmp[-1]
            _boxes_Flidar.append(tmp.reshape(1, -1))
        return np.concatenate(_boxes_Flidar, axis=0)
    # compute riou to associate boxes and qboxes
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    # update the non-zero values in riou with JIoUs
    ## transform boxes and qboxes to LiDAR frame
    boxes_Flidar = _transform(boxes, calibs)
    qboxes_Flidar = _transform(qboxes, qcalibs)
    ## compute jious
    for box_idx, box_qboxes_riou in enumerate(riou):
        qboxes_idx = np.where(box_qboxes_riou>0)[0]
        # if is DontCare, pass it
        if dcmasks[box_idx]:
            continue
        # filter out meaning less items in qboxes_idx
        # - DontCare labels
        # - other frame labels
        dc_and_cross_label_filter = lambda qbox_idx: \
            (not qdcmasks[qbox_idx]) and \
            (qframeids[qbox_idx] == frameids[box_idx])
        qboxes_idx = list(filter(dc_and_cross_label_filter, qboxes_idx))
        if len(qboxes_idx) == 0:
            continue
        if not bool_delta:
            ucts_ = [ucts[box_idx]]
            qucts_ = [qucts[itm] for itm in qboxes_idx]
        elif not is_boxes_gt:
            ucts_ = [ucts[box_idx]]
            qucts_ = [{"mean": itm,
                       "cov": np.diag(np.zeros_like(itm) + 1e-8)}
                      for itm in qboxes_Flidar[qboxes_idx].reshape(-1, 5)]
        elif is_boxes_gt:
            ucts_ = [{"mean": itm,
                      "cov": np.diag(np.zeros_like(itm) + 1e-8)}
                     for itm in boxes_Flidar[qboxes_idx].reshape(-1, 5)]
            qucts_ = [qucts[itm] for itm in qboxes_idx]
        jiou = jiou_eval_bev(
            boxes_Flidar[box_idx].reshape(-1, 5), ucts_,
            qboxes_Flidar[qboxes_idx].reshape(-1, 5), qucts_,
            3.0, 0.1, use_mean=True)
        riou[box_idx, qboxes_idx] = jiou
    return riou

def d3_box_overlap_uct(
    boxes:np.ndarray, qboxes:np.ndarray,
    ucts:Dict[str,np.ndarray], qucts:Dict[str,np.ndarray],
    calibs:List, qcalibs:List,
    dcmasks:List[bool], qdcmasks:List[bool],
    frameids:List[str], qframeids:List[str],
    criterion:int=-1,
    bool_delta:bool=False, is_boxes_gt:bool=False) -> np.ndarray:
    '''
    Evaluate overlap between boxes and qboxes considering the uncertainties.
    Args:
        boxes (N, 7): 3D bounding boxes (kitti format) (in left camera frame)
        qboxes (M, 7): 3D bounding boxes (kitti format) (in left camera frame) (query)
        ucts (N): predictive distributions (in Lidar frame) . Each item contains a dict:
            {"mean": np.ndarray (1,8)/(1,7),
             "cov": np.ndarray (8,8)/(7,7)}
        qucts (M): predictive distributions (in Lidar frame) (query). Each item contains a dict:
            {"mean": np.ndarray (1,8)/(1,7),
             "cov": np.ndarray (8,8)/(7,7)}
        calibs (List[KittiCalib])(N): calib classes
        qcalibs (List[KittiCalib])(M): calib classes (query)
        dcmasks (N): it indicates whether it is a 'DontCare' class.
        qdcmasks (M): it indicates whether it is a 'DontCare' class. (query)
        frameids (N): frame ids
        qfameids (M): frame ids (query)
        criterion:
        bool_delta (bool): if True, make gts as a delta distribution
        is_boxes_gt (bool): True indicates <boxes> are labels. Otherwise
         it indicates <qboxes> are labels. It will take effect in computing delta JIoU.
    Returns:
        riou (N, M): IoU matrix
    TODO: This function can be further improved:
        - if it computes JIoU in the camera frame;
    '''
    def _transform(_boxes, _calibs):
        _boxes_Flidar = []
        for box, calib in zip(_boxes, _calibs):
            x, y, z, l, h, w, ry = box.reshape(-1)
            bottom_Fcam = np.array([[x, y, z]])
            bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
            bbox3d = np.array([
                bottom_Flidar[0,0],
                bottom_Flidar[0,1],
                bottom_Flidar[0,2]+h/2.0, # center
                l, w, h,
                np.pi/2-ry]).reshape(1,7)
            _boxes_Flidar.append(bbox3d)
        return np.concatenate(_boxes_Flidar, axis=0)
    # compute riou to associate boxes and qboxes
    riou = d3_box_overlap(boxes, qboxes, criterion)
    # update the non-zero values in riou with JIoUs
    ## transform boxes and qboxes to LiDAR frame
    boxes3d_Flidar = _transform(boxes, calibs)
    qboxes3d_Flidar = _transform(qboxes, qcalibs)
    ## compute jious
    for box_idx, box_qboxes_riou in enumerate(riou):
        qboxes_idx = np.where(box_qboxes_riou>0)[0]
        # if is DontCare, pass it
        if dcmasks[box_idx]:
            continue
        # filter out meaning less items in qboxes_idx
        # - DontCare labels
        # - other frame labels
        dc_and_cross_label_filter = lambda qbox_idx: \
            (not qdcmasks[qbox_idx]) and \
            (qframeids[qbox_idx] == frameids[box_idx])
        qboxes_idx = list(filter(dc_and_cross_label_filter, qboxes_idx))
        if len(qboxes_idx) == 0:
            continue
        if not bool_delta:
            ucts_ = [ucts[box_idx]]
            qucts_ = [qucts[itm] for itm in qboxes_idx]
        elif not is_boxes_gt:
            ucts_ = [ucts[box_idx]]
            qucts_ = [{"mean": itm,
                       "cov": np.diag(np.zeros_like(itm) + 1e-8)}
                      for itm in qboxes3d_Flidar[qboxes_idx].reshape(-1, 7)]
        elif is_boxes_gt:
            ucts_ = [{"mean": itm,
                      "cov": np.diag(np.zeros_like(itm) + 1e-8)}
                     for itm in boxes3d_Flidar[qboxes_idx].reshape(-1, 7)]
            qucts_ = [qucts[itm] for itm in qboxes_idx]

        jiou = jiou_eval_3d(
            boxes3d_Flidar[box_idx].reshape(-1, 7),
            ucts_,
            qboxes3d_Flidar[qboxes_idx].reshape(-1, 7),
            qucts_,
            3.0, 0.1, use_mean=True)
        riou[box_idx, qboxes_idx] = jiou
    return riou

def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50, bool_uct=False):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    def _get_kwargs(gt_annos_part, dt_annos_part, mode):
        assert mode in ['bev', '3d']
        gt_ucts = [[{
            "mean": itm[f"post_mean_{mode}"],
            "cov": itm[f"post_cov_{mode}"]}
            for itm in a['uncertainty']]
            for a in gt_annos_part]
        gt_ucts = list(chain.from_iterable(gt_ucts))
        gt_calibs = list(chain.from_iterable(
            [a['calib'] for a in gt_annos_part]))
        gt_dcmasks = list(chain.from_iterable(
            [a['name']=='DontCare' for a in gt_annos_part \
            if a['name'].shape[0]>0])) # the empty anno will cause an error
        gt_frameids = list(chain.from_iterable(
            [a['frameid'] for a in gt_annos_part]))
        dt_ucts = [[{
            "mean": itm[f"post_mean_{mode}"],
            "cov": itm[f"post_cov_{mode}"]}
            for itm in a['uncertainty']]
            for a in dt_annos_part]
        dt_ucts = list(chain.from_iterable(dt_ucts))
        dt_calibs = list(chain.from_iterable(
            [a['calib'] for a in dt_annos_part]))
        dt_dcmasks = list(chain.from_iterable(
            [a['name']=='DontCare' for a in dt_annos_part \
            if a['name'].shape[0]>0]))
        dt_frameids = list(chain.from_iterable(
            [a['frameid'] for a in dt_annos_part]))
        return dict(
            boxes=gt_boxes, qboxes=dt_boxes,
            ucts=gt_ucts, qucts=dt_ucts,
            calibs=gt_calibs, qcalibs=dt_calibs,
            dcmasks=gt_dcmasks, qdcmasks=dt_dcmasks,
            frameids=gt_frameids, qframeids=dt_frameids)
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            if bool_uct:
                bev_box_overlap_uct_kwargs = _get_kwargs(
                    gt_annos_part, dt_annos_part, mode='bev')
                with Timer(s="bev_box_overlap"):
                    overlap_part = bev_box_overlap_uct(
                    **bev_box_overlap_uct_kwargs,
                    bool_delta=False, is_boxes_gt=False).astype(np.float64)
            else:
                overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                    np.float64)
        elif metric == 2:
            # xc_cam, yc_cam, zc_cam
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            # l, h, w
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            # ry
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            # xc_cam, yc_cam, zc_cam
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            # l, h, wc
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            # ry
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            if bool_uct:
                d3_box_overlap_uct_kwargs = _get_kwargs(
                    gt_annos_part, dt_annos_part, mode='3d')
                with Timer(s="d3_box_overlap"):
                    overlap_part = d3_box_overlap_uct(
                    **d3_box_overlap_uct_kwargs,
                    bool_delta=False, is_boxes_gt=False).astype(np.float64)
            else:
                overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None, bool_uct=False):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict, bool_uct=bool_uct)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))

            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                if i == 0:
                   ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                   ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                   ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            if i == 0:
                # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
                # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
                # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
                # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict

def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None,
            bool_uct=False):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos, bool_uct=bool_uct)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps, bool_uct=bool_uct)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, bool_uct=bool_uct)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40

def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100,
               bool_uct=False):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm
        bool_uct: bool, if True, evaluate with conrideration of uncertainties.
    Returns:
        dict of recall, precision and aos
    TODO: swap the dt_annos and gt_annos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts, bool_uct=bool_uct)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict
