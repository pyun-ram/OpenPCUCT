'''
 File Created: Mon Mar 02 2020

'''
import torch
import numpy as np
from det3.utils.utils import rotz, apply_R, apply_tr

def get_corner_box_2drot_np(boxes):
    '''
    get corners of 2d rotated boxes.
    @boxes: np.ndarray (M, 5)
        [[x, y, l, w, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
        2.--.3
         |  |                  ^x
         |  |                  |
        1.--.4 (bottom) y<----.z
    -> corners: np.ndarray (M, 4, 2)
    Note: 35 ms for 100 times (30 boxes)
    '''
    M = boxes.shape[0]
    l = boxes[:, 2:3]
    w = boxes[:, 3:4]
    theta = boxes[:, 4:5]
    p1 = np.hstack((-l/2.0, w/2.0, np.zeros_like(l)))
    p2 = np.hstack(( l/2.0, w/2.0, np.zeros_like(l)))
    p3 = np.hstack(( l/2.0,-w/2.0, np.zeros_like(l)))
    p4 = np.hstack((-l/2.0,-w/2.0, np.zeros_like(l)))
    pts = np.stack((p1, p2, p3, p4), axis=0)
    pts = np.transpose(pts, (1, 0, 2))
    tr_vecs = boxes[:, :2]
    tr_vecs = np.hstack((tr_vecs, np.zeros((M, 1))))
    for i, (tr_, ry_) in enumerate(zip(tr_vecs, theta)):
        pts[i] = apply_R(pts[i], rotz(ry_))
        pts[i] = apply_tr(pts[i], tr_)
    return pts[:, :, :2]

def get_corner_box_2drot_torch(boxes):
    '''
    get corners of 2d rotated boxes.
    @boxes: torch.Tensor/ torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
        2.--.3
         |  |                  ^x
         |  |                  |
        1.--.4 (bottom) y<----.z
    -> corners: torch.Tensor/ torch.Tensor.cuda (M, 4, 2)
    Note: 10 ms for 100 times (30 boxes) (cpu)
          30 ms for 100 times (30 boxes) (gpu w.o. transfering time)
    '''
    # device = boxes.device
    # dtype = boxes.dtype
    # M = boxes.shape[0]
    # l = boxes[:, 2:3]
    # w = boxes[:, 3:4]
    # theta = boxes[:, 4]
    # p1 = torch.stack([-l/2.0, w/2.0], dim=1)
    # p2 = torch.stack([ l/2.0, w/2.0], dim=1)
    # p3 = torch.stack([ l/2.0,-w/2.0], dim=1)
    # p4 = torch.stack([-l/2.0,-w/2.0], dim=1)
    # pts = torch.stack([p1, p2, p3, p4], dim=0)
    # pts = pts.transpose(0, 1).squeeze()
    # tr_vecs = boxes[:, :2].unsqueeze(1)
    # tr_vecs = tr_vecs.repeat(1,4,1)
    # R = torch.eye(2, device=device,
    #     dtype=dtype).repeat([M, 1, 1])
    # cry = torch.cos(theta)
    # sry = torch.sin(theta)
    # R[:, 0, 0] = cry
    # R[:, 1, 1] = cry
    # R[:, 0, 1] = -sry
    # R[:, 1, 0] = sry
    # pts = torch.bmm(R, pts.transpose(-2, -1)).transpose(-2, -1)
    # pts += tr_vecs
    import boxop_cpp
    return boxop_cpp.get_corner_box_2drot(boxes)

def get_corner_box_3drot_np(boxes):
    '''
    get corners of 3d rotated boxes.
    @boxes: np.ndarray (M, 7)
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
        8.--.7 (top)
    -> corners: np.ndarray (M, 8, 3)
    Note: 35 ms for 100 times (30 boxes)
    '''
    # M, 4, 2
    cns2d = get_corner_box_2drot_np(boxes[:, [0,1,3,4,6]])
    # M, 4
    z = np.tile(boxes[:, 2:3], (1, 4))
    z = np.expand_dims(z, -1)
    h = np.tile(boxes[:, 5:6], (1, 4))
    cns3d_bottom = np.concatenate([cns2d, z], axis=-1)
    cns3d_top = cns3d_bottom.copy()
    cns3d_top[:, :, -1] += h
    return np.concatenate([cns3d_bottom, cns3d_top], axis=1)

def get_corner_box_3drot_torch(boxes):
    '''
    get corners of 3d rotated boxes.
    @boxes: torch.Tensor/ torch.Tensor.cuda (M, 7)
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
        8.--.7 (top)
    -> corners: torch.Tensor/ torch.Tensor.cuda (M, 8, 3)
    Note: 18 ms for 100 times (30 boxes) (cpu)
          48 ms for 100 times (30 boxes) (gpu w.o. transfering time)
    '''
    # M, 4, 2
    cns2d = get_corner_box_2drot_torch(boxes[:, [0,1,3,4,6]])
    # M, 4
    z = boxes[:, 2:3].repeat(1, 4).unsqueeze(-1)
    h = boxes[:, 5:6].repeat(1, 4)
    cns3d_bottom = torch.cat([cns2d, z], dim=-1)
    cns3d_top = cns3d_bottom.clone()
    cns3d_top[:, :, -1] += h
    return torch.cat([cns3d_bottom, cns3d_top], dim=1)

def crop_pts_3drot_np(boxes, pts):
    '''
    crop pts acoording to the 3D boxes.
    @boxes: np.ndarray (M, 7)
            [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
            l, w, and h are the scales along x-, y-, and z- axes.
            theta is the rotation angle along the z-axis (counter-clockwise).
    @pts: np.ndarray (N, 3)
    [[x, y, z]...]
    -> idxes: list [np.long, ...]
        assert len (idxes) == M
    '''
    res =  crop_pts_3drot_torchgpu(torch.from_numpy(boxes).cuda(),
        torch.from_numpy(pts).cuda())
    return [itm.cpu().numpy() for itm in res]

def crop_pts_3drot_torchcpu(boxes, pts):
    '''
    crop pts acoording to the 3D boxes.
    @boxes: torch.Tensor (M, 7)
            [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
            l, w, and h are the scales along x-, y-, and z- axes.
            theta is the rotation angle along the z-axis (counter-clockwise).
    @pts: torch.Tensor (N, 3)
    [[x, y, z]...]
    -> idxes: list [torch.long, ...]
        assert len (idxes) == M
    '''
    res =  crop_pts_3drot_torchgpu(boxes.cuda(), pts.cuda())
    return [itm.cpu() for itm in res]

def crop_pts_3drot_torchgpu(boxes, pts):
    '''
    crop pts acoording to the 3D boxes.
    @boxes: torch.Tensor.cuda (M, 7)
            [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
            l, w, and h are the scales along x-, y-, and z- axes.
            theta is the rotation angle along the z-axis (counter-clockwise).
    @pts: torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    -> idxes: list [torch.long.cuda, ...]
        assert len (idxes) == M
    '''
    import boxop_cuda
    res = boxop_cuda.crop_pts_3drot(boxes, pts)
    res = [torch.nonzero(mask, as_tuple=False).flatten() for mask in res]
    return res