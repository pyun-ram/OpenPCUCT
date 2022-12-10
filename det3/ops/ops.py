'''
 File Created: Sat Feb 29 2020

 Note: This file includes the APIs of det3 operations.
    All the functions have to be well documented.
    The document of each API should include:
    - function
    - input (type) description
    - output (type) description
    - Special Note: If not specified, the function should be safe.
    i.e. not changing the inputs.
'''
import torch
import numpy as np
from det3.utils.utils import rotx, roty, rotz
# I/O
def read_txt(path:str):
    from det3.ops.io import read_txt_
    return read_txt_(path)

def write_txt(obj:list, path:str):
    from det3.ops.io import write_txt_
    write_txt_(obj, path)

def read_npy(path:str):
    from det3.ops.io import read_npy_
    return read_npy_(path)

def write_npy(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    from det3.ops.io import write_npy_
    write_npy_(obj, path)

def read_pcd(path:str):
    from det3.ops.io import read_pcd_
    return read_pcd_(path)

def write_pcd(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    from det3.ops.io import write_pcd_
    write_pcd_(obj, path)

def read_bin(path:str, dtype):
    '''
    @dtype: np.float32/np.float64
    '''
    from det3.ops.io import read_bin_
    return read_bin_(path, dtype)

def write_bin(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    from det3.ops.io import write_bin_
    write_bin_(obj, path)

def read_img(path:str):
    from det3.ops.io import read_img_
    return read_img_(path)

def write_img(obj, path:str):
    from det3.ops.io import write_img_
    write_img_(obj, path)

def read_pkl(path:str):
    from det3.ops.io import read_pkl_
    return read_pkl_(path)

def write_pkl(obj, path:str):
    '''
    @obj: any python object
    '''
    from det3.ops.io import write_pkl_
    write_pkl_(obj, path)

def read_json(path:str):
    from det3.ops.io import read_json_
    return read_json_(path)

# IoU Computing
def compute_intersect_2d(boxes, others):
    '''
    compute the intersection between box and others under 2D aligned boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 4)
        [[x, y, l, w],...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M', 4)
        [[x, y, l, w],...]
    -> its: intersection results with same type as boxes (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.05ms
        torch.Tensor: 1.5ms
        torch.Tensor.cuda: 4ms
    '''
    from det3.ops.iou import (compute_intersect_2d_npy,
        compute_intersect_2d_torch)
    if isinstance(boxes, np.ndarray):
        return compute_intersect_2d_npy(boxes, others).astype(boxes.dtype)
    elif isinstance(boxes, torch.Tensor):
        return compute_intersect_2d_torch(boxes, others)
    else:
        raise NotImplementedError

def compute_iou_2d(boxes, others):
    '''
    compute the iou between box and others under 2D aligned boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 4)
        [[x, y, l, w],...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M', 4)
        [[x, y, l, w],...]
    -> iou: iou results with same type as boxes (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.05ms
        torch.Tensor: 1.5ms
        torch.Tensor.cuda: 4ms
    '''
    area1 = compute_area( boxes[:, [2, 3]])
    area2 = compute_area(others[:, [2, 3]])
    inter = compute_intersect_2d(boxes, others)
    return compute_iou(inter, area1, area2)

def compute_intersect_2drot(boxes, others):
    '''
    compute the intersection between box and others under 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta],...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M', 5)
        [[x, y, l, w, theta],...]
    -> its: intersection results with same type as box (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.3ms
        torch.Tensor: 0.3ms
        torch.Tensor.cuda: 0.2ms
    '''
    from det3.ops.iou import (compute_intersect_2drot_npy,
        compute_intersect_2drot_torchcpu, compute_intersect_2drot_torchgpu)
    if isinstance(boxes, np.ndarray):
        return compute_intersect_2drot_npy(boxes, others)
    elif isinstance(boxes, torch.Tensor) and not boxes.is_cuda:
        return compute_intersect_2drot_torchcpu(boxes, others)
    elif isinstance(boxes, torch.Tensor) and boxes.is_cuda:
        return compute_intersect_2drot_torchgpu(boxes, others)
    else:
        raise NotImplementedError

def compute_iou_2drot(boxes, others):
    '''
    compute the iou between box and others under 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta],...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M', 5)
        [[x, y, l, w, theta],...]
    -> iou: iou results with same type as box (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.3ms
        torch.Tensor: 0.3ms
        torch.Tensor.cuda: 0.2ms
    '''
    area1 = compute_area( boxes[:, [2, 3]])
    area2 = compute_area(others[:, [2, 3]])
    inter = compute_intersect_2drot(boxes, others)
    return compute_iou(inter, area1, area2)

def compute_intersect_3drot(boxes, others):
    '''
    compute the intersection between boxes and others under 3D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
        [x, y, z, l, w, h, theta] (x, y, z) is the bottom center coordinate;
        l, w, and h are the scales along x-, y-, and z- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M', 7)
        [[x, y, z, l, w, h, theta],...]
    -> its: intersection results with same type as box (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.5ms
        torch.Tensor: 0.7ms
        torch.Tensor.cuda: 0.8ms
    '''
    from det3.ops.iou import compute_intersect_3drot_np, compute_intersect_3drot_torch
    intersec_2drot = compute_intersect_2drot(boxes[:, [0, 1, 3, 4, 6]],
                                             others[:, [0, 1, 3, 4, 6]])
    if isinstance(boxes, np.ndarray):
        return compute_intersect_3drot_np(boxes, others, intersec_2drot)
    elif isinstance(boxes, torch.Tensor):
        return compute_intersect_3drot_torch(boxes, others, intersec_2drot)
    else:
        raise NotImplementedError

def compute_iou_3drot(boxes, others):
    '''
    compute the iou between boxes and others under 3D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
        [x, y, z, l, w, h, theta] (x, y, z) is the bottom center coordinate;
        l, w, and h are the scales along x-, y-, and z- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M', 7)
        [[x, y, z, l, w, h, theta],...]
    -> iou: iou results with same type as box (M, M')
    %time: M = 10; M' = 21;
        np.ndarray: 0.5ms
        torch.Tensor: 0.7ms
        torch.Tensor.cuda: 0.8ms
    '''
    vol1 = compute_volume( boxes[:, 3:6])
    vol2 = compute_volume(others[:, 3:6])
    inter = compute_intersect_3drot(boxes, others)
    return compute_iou(inter, vol1, vol2)

def compute_area(boxes):
    '''
    compute the area of rectangles
    @boxes: np.ndarray/ torch.Tensor/ torch.Tensor.cuda (M, 2)
    ->areas: (M,)
    '''
    return boxes[:, 0] * boxes[:, 1]


def compute_volume(boxes):
    '''
    compute the volume of cubes
    @boxes: np.ndarray/ torch.Tensor/ torch.Tensor.cuda (M, 3)
    ->vols: (M,)
    '''
    return boxes[:, 0] * boxes[:, 1] * boxes[:, 2]

def compute_iou(inter, area1, area2):
    '''
    compute the iou, given the inter, area1, area2 (2d)
    compute the iou, given the inter, vol1, vol2 (2d)
    @inter: the intersection np.ndarray/ torch.Tensor/ torch.Tensor.cuda (M, M')
    @area1: area1/vol1 np.ndarray/ torch.Tensor/ torch.Tensor.cuda (M,)
    @area2: area2/vol2 np.ndarray/ torch.Tensor/ torch.Tensor.cuda (M',)
    -> ious: same to inter (M, M')
    '''
    M = area1.shape[0]
    M_ = area2.shape[0]
    assert inter.shape[0] == M
    assert inter.shape[1] == M_
    if isinstance(inter, np.ndarray):
        tmp1 = area1.repeat(M_).reshape(M, M_)
        tmp2 = area2.repeat(M).reshape(M_, M).T
        tmp = tmp1 + tmp2
    elif isinstance(inter, torch.Tensor):
        tmp1 = area1.repeat_interleave(M_).reshape(M, M_)
        tmp2 = area2.repeat_interleave(M).reshape(M_, M).transpose(0, 1)
        tmp = tmp1 + tmp2
    else:
        raise NotImplementedError
    return inter / (tmp - inter)

# NMS
def nms_2d(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D aligned boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 4)
        [[x, y, l, w]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

def nms_2drot(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

def nms_3drot(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
        [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
        l, w, and h are the scales along x-, y-, and z- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

# Point Operation
def apply_tr(pts, tr_vec):
    '''
    apply translation on pts.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
        [[x, y, z]...]
    @tr_vec: same to pts (3, )
        [tr_x, tr_y, tr_z]
    -> pts_tr same to pts (N, 3)
    Note: This function should be safe.
    '''
    return pts + tr_vec

def apply_R(pts, R_matrix):
    '''
    apply rotation on pts.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
        [[x, y, z]...]
    @R_matrix: same to pts (3, 3)
        3x3 rotation matrix
    -> pts_R same to pts (N, 3)
    Note: This function should be safe.
    '''
    return (R_matrix @ pts.T).T

def apply_T(pts, T_matrix):
    '''
    apply transformation matrix T on pts.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
        [[x, y, z]...]
    @T_matrix: same to pts (4, 4)
        3x3 rotation matrix
    -> pts_R same to pts (N, 3)
    Note: This function should be safe.
    '''
    pts_h = hfill_pts(pts)
    pts_ = (T_matrix @ pts_h.T).T
    return pts_[:, :3]

def hfill_pts(pts):
    '''
    convert pts to homogeneous coordinate.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    -> pts_h same to pts (N, 4)
    [[x, y, z, 1]...]
    '''
    if isinstance(pts, np.ndarray):
        return np.hstack([pts[:, :3], np.ones_like(pts[:, 0:1])])
    else:
        raise NotImplementedError

def farthest_pts_sampling(pts, num_pts: int):
    '''
    farthest points sampling.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    @num_pts: num of points in the returned value
    -> idx: same to pts (num_pts, ) dtype=long
    '''
    raise NotImplementedError

# Box Operation
def get_corner_box_2drot(boxes):
    '''
    get corners of 2d rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
        1.--.2
         |  |                  ^x
         |  |                  |
        4.--.3 (bottom) y<----.z
    -> corners: same to boxes (M, 4, 2)
    %time M == 30
    np 0.4 ms
    torchcpu 0.2 ms
    torchgpu 0.4 ms
    '''
    from det3.ops.boxop import (get_corner_box_2drot_np,
        get_corner_box_2drot_torch)
    if isinstance(boxes, np.ndarray):
        return get_corner_box_2drot_np(boxes)
    elif isinstance(boxes, torch.Tensor):
        return get_corner_box_2drot_torch(boxes)
    else:
        raise NotImplementedError

def get_corner_box_3drot(boxes):
    '''
    get corners of 3d rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
        [[x, y, z, l, w, h, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
        1.--.2
         |  |                  ^x
         |  |                  |
        4.--.3 (bottom) y<----.z
        5.--.6
         |  |
         |  |
        8.--.7 (bottom)
    -> corners: same to boxes (M, 4, 3)
    %time M == 30
    np 0.4 ms
    torchcpu 0.2 ms
    torchgpu 0.4 ms
    '''
    from det3.ops.boxop import (get_corner_box_3drot_np,
        get_corner_box_3drot_torch)
    if isinstance(boxes, np.ndarray):
        return get_corner_box_3drot_np(boxes)
    elif isinstance(boxes, torch.Tensor):
        return get_corner_box_3drot_torch(boxes)
    else:
        raise NotImplementedError

def crop_pts_3drot(boxes, pts):
    '''
    crop pts acoording to the 3D boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
            [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
            l, w, and h are the scales along x-, y-, and z- axes.
            theta is the rotation angle along the z-axis (counter-clockwise).
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    -> idxes: list [np.long/torch.long/torch.long.cuda, ...]
        assert len (idxes) == M
    %time M == 5
    np 0.4 ms
    torchcpu 0.4 ms
    torchgpu 0.3 ms
    '''
    from det3.ops.boxop import (crop_pts_3drot_np,
        crop_pts_3drot_torchcpu, crop_pts_3drot_torchgpu)
    if isinstance(boxes, np.ndarray):
        return crop_pts_3drot_np(boxes, pts)
    elif isinstance(boxes, torch.Tensor) and not boxes.is_cuda:
        return crop_pts_3drot_torchcpu(boxes, pts)
    elif isinstance(boxes, torch.Tensor) and boxes.is_cuda:
        return crop_pts_3drot_torchgpu(boxes, pts)
    else:
        raise NotImplementedError
