'''
File Created: Sunday, 17th March 2019 9:41:35 pm

'''
import numpy as np
from PIL import Image
import pickle

def is_param(key: str) -> bool:
    '''if key is start with @, then it returns True'''
    return key[0] == '@'

def proc_param(key: str) -> str:
    '''remove the @ of key'''
    assert is_param(key), f"{key} is not a valid key starting with '@'"
    return key[1:]

def write_pc_to_file(pc, path):
    '''
    @ pc: np.array (np.float32)
    '''
    assert pc.dtype.type is np.float32
    ftype = path.split(".")[-1]
    if ftype == "bin":
        with open(path, 'wb') as f:
            pc.tofile(f)
    elif ftype == "npy":
        np.save(path, pc)
    elif ftype == "pcd":
        import open3d
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pc[:, :3])
        open3d.io.write_point_cloud(path, pcd)
    else:
        print(ftype)
        raise NotImplementedError

def read_pc_from_file(path, num_feature=4):
    ftype = path.split(".")[-1]
    if ftype == "bin":
        return read_pc_from_bin(path, num_feature=num_feature)
    elif ftype == "npy":
        return read_pc_from_npy(path)
    else:
        print(ftype)
        raise NotImplementedError

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def write_str_to_file(s, file_path):
    with open(file_path, 'w+') as f:
        f.write(s)

def get_idx_list(txt_path):
    '''
    get idx from the txt
    inputs:
        txt_path(str): the txt path
    '''
    idx_list = []
    with open(txt_path, 'r') as f:
        idx_list = f.readlines()
    return [itm.rstrip() for itm in idx_list]

def read_image(path):
    '''
    read image
    inputs:
        path(str): image path
    returns:
        img(np.array): [w,h,c]
    '''
    return np.array(Image.open(path, 'r'))

# BUG HERE
# def read_pc_from_pcd(pcd_path):
#     """Load PointCloud data from pcd file."""
#     from open3d import read_point_cloud
#     pcd = read_point_cloud(pcd_path)
#     pc = np.asarray(pcd.points)
#     return pc

# def read_pc_from_ply(ply_path):
#     '''Load PointCloud data from ply file'''
#     from open3d import read_point_cloud
#     pcd = read_point_cloud(ply_path)
#     pc = np.asarray(pcd.points)
#     return pc

def read_pc_from_npy(npy_path):
    """Load PointCloud data from npy file."""
    p = np.load(npy_path)
    assert not np.isnan(np.min(p))
    return p

def read_pc_from_bin(bin_path, num_feature=4):
    """Load PointCloud data from bin file."""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, num_feature)
    return p

def rotx(t):
    ''' 3D Rotation about the x-axis.
    source: https://github.com/charlesq34/pointnet
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]]).astype(float)

def roty(t):
    ''' Rotation about the y-axis.
    source: https://github.com/charlesq34/pointnet'''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]]).astype(float)

def rotz(t):
    ''' Rotation about the z-axis.
    source: https://github.com/charlesq34/pointnet'''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]]).astype(float)

def apply_R(pts, R):
    '''
    apply Rotation Matrix on pts
    inputs:
        pts (np.array): [#pts, 3]
        R (np.array): [3,3]
            Rotation Matix
        Note: pts and R should match with each other.
    return:
        pts_R (np.array): [#pts, 3]
    '''
    return (R @ pts.T).T

def apply_tr(pts, tr):
    '''
    apply Translation Vector on pts
    inputs:
        pts (np.array): [#pts, 3]
        tr (np.array): [1, 3]
        Note: pts and tr should match with each other.
    return:
        pts_tr (np.array): [#pts, 3]
    '''
    return pts + tr

def clip_ry(ry):
    '''
    clip ry to [-pi..pi] range
    inputs: 
        ry (float)
    '''
    while ry <= -np.pi:
        ry += np.pi
    while ry >= np.pi:
        ry -= np.pi
    return ry

def istype(obj, type_name):
    '''
    check if obj is class <type_name>
    inputs:
        obj: object
        type_name: str
    return:
        True is obj is class <type_name>
    '''
    return obj.__class__.__name__ is type_name

def nms_general(boxes, scores, threshold, mode='2d-rot'):
    '''
    Non-Maximum Surpression (NMS).
    inputs:
        boxes (np.array) [# boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
        scores (np.array) [# boxes, ]
        threshold (float): keep the boxes with the IoU <= thresold
        mode (str): '2d', '2d-rot', '3d-rot'
    return:
        idx (list)
    '''
    # sort in descending order of scores
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i) # keep the one with highest score
        ovr = compute_iou(boxes[i:i+1], boxes[order[1:]], mode=mode) # compute iou with others
        inds = np.where(ovr <= threshold)[0] # keep the boxes with the IoU <= thresold
        order = order[inds + 1]
    return keep

def compute_iou(box, others, mode='2d-rot'):
    '''
    compute IoU between box and others.
    inputs:
        box (np.array) [1, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
        others (np.array) [#boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
        mode (str): '2d-rot', '3d-rot'
    return:
        ious(np.array) [#boxes,]
    '''
    if mode == '2d-rot':
        # compute box area
        box_area = compute_area(box)
        # compute others area
        others_area = compute_area(others)
        # compute intersection
        inter = compute_intersec(box, others, mode)
        # compute iou
        iou = inter / (box_area + others_area - inter)
    elif mode == '3d-rot':
        # compute box volume
        box_vol = compute_volume(box)
        # compute others volume
        others_vol = compute_volume(others)
        # compute intersection
        inter = compute_intersec(box, others, mode)
        # compute iou
        iou = inter / (box_vol + others_vol - inter)
    else:
        raise NotImplementedError
    return iou

def compute_area(boxes):
    '''
    compute area of boxes.
    inputs:
        boxes (np.array) [#boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
    return:
        areas (np.array) [#boxes, ]
    '''
    return boxes[:, 3] * boxes[:, 4]

def compute_volume(boxes):
    '''
    compute area of boxes.
    inputs:
        boxes (np.array) [#boxes, 7]
        3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
    return:
        vols (np.array) [#boxes, ]
    '''
    return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

def compute_intersec(box, others, mode):
    '''
    compute intersect between box and others.
    inputs:
        box (np.array) [1, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
        others (np.array) [#boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
        mode (str): '2d-rot', '3d-rot'
    return:
        inter(np.array) [#boxes,]
    '''
    # from shapely.geometry import Polygon
    num_all = others.shape[0]+1 # total number of all boxes (box + other)
    all_boxes = np.vstack([box, others]) # num_all, 7 (x, y, z, l, w, h, ry)
    all_plg2d = boxes3d2polygon(all_boxes) # convert boxes3d into 2d polygons for intersection computing
    if mode == '2d-rot':
        inter = np.zeros(others.shape[0])
        for i, plg in enumerate(all_plg2d[1:]):
            inter[i] = all_plg2d[0].intersection(plg).area
    elif mode == '3d-rot':
        min_h_box = all_boxes[0, 2]
        max_h_box = all_boxes[0, 2] + all_boxes[0, 5]
        min_h_others = all_boxes[1:, 2]
        max_h_others = all_boxes[1:, 2] + all_boxes[1:, 5]
        max_of_min = np.maximum(min_h_box, min_h_others)
        min_of_max = np.minimum(max_h_box, max_h_others)
        inter_z = np.maximum(0, min_of_max - max_of_min)
        inter_xy = np.zeros(others.shape[0])
        for i, plg in enumerate(all_plg2d[1:]):
            inter_xy[i] = all_plg2d[0].intersection(plg).area
        inter = inter_xy * inter_z
    else:
        raise NotImplementedError
    return inter

def boxes3d2polygon(boxes):
    '''
    convert boxes defined by [x, y, z, l, w, h, ry] into polygons
    inputs:
        boxes: (#boxes, 7) [x, y, z, l, w, h, ry]
               ^y
               |
            (z).----->x
            bottom center :[x,y,z]
            l, w, h <-> x, y, z
            ry : rotation along z axis
    outputs:
        polygons (list) [#boxes]
    '''
    from shapely.geometry import Polygon
    num_all = boxes.shape[0]
    all_cns = np.zeros([num_all, 8, 3])
    all_x = boxes[:, 0:1]
    all_y = boxes[:, 1:2]
    all_z = boxes[:, 2:3]
    all_l = boxes[:, 3:4]
    all_w = boxes[:, 4:5]
    all_h = boxes[:, 5:6]
    all_ry = boxes[:, 6:7]
    array0 = np.zeros([num_all, 1])
    all_cns[:, 0, :] = np.concatenate([-all_l/2.0, all_w/2.0, array0], axis=1)
    all_cns[:, 1, :] = np.concatenate([ all_l/2.0, all_w/2.0, array0], axis=1)
    all_cns[:, 2, :] = np.concatenate([ all_l/2.0,-all_w/2.0, array0], axis=1)
    all_cns[:, 3, :] = np.concatenate([-all_l/2.0,-all_w/2.0, array0], axis=1)
    all_cns[:, 4, :] = all_cns[:, 0, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 5, :] = all_cns[:, 1, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 6, :] = all_cns[:, 2, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 7, :] = all_cns[:, 3, :] + np.concatenate([array0, array0, all_h], axis=1)
    for i in range(all_cns.shape[0]):
        all_cns[i, :, :] = apply_R(all_cns[i, :, :], rotz(all_ry[i]))
        all_cns[i, :, :] = apply_tr(all_cns[i, :, :], np.array([all_x[i], all_y[i], all_z[i]]).reshape(1, 3))
    all_cns2d = all_cns[:, :4, :2]
    all_plg2d = []
    for cns in all_cns:
        all_plg2d.append(Polygon(cns.tolist()).buffer(0)) # buffer is to clean bowties
    return all_plg2d

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count