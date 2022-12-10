'''
File Created: Sunday, 17th March 2019 3:58:52 pm

'''
import os
import math
import numpy as np
from numpy.linalg import inv
from enum import Enum
try:
    from ..utils import utils
except:
    # Run script python3 dataloader/kittidata.py
    import det3.utils.utils as utils

Frame = Enum('Frame', ('LiDAR', 'Cam2'))

# KITTI
class KittiCalib:
    '''
    class storing KITTI calib data
        self.data(None/dict):keys: 'P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'
        self.R0_rect(np.array):  [4,4]
        self.Tr_velo_to_cam(np.array):  [4,4]
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None
        self.R0_rect = None
        self.Tr_velo_to_cam = None

    def read_calib_file(self):
        '''
        read KITTI calib file
        '''
        calib = dict()
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        self.data = calib

        R0_rect = np.zeros([4, 4])
        R0_rect[0:3, 0:3] = np.array(self.data['R0_rect']).reshape(3, 3)
        R0_rect[3, 3] = 1
        self.R0_rect = R0_rect

        Tr_velo_to_cam = np.zeros([4, 4])
        Tr_velo_to_cam[0:3, :] = np.array(self.data['Tr_velo_to_cam']).reshape(3, 4)
        Tr_velo_to_cam[3, 3] = 1
        self.Tr_velo_to_cam = Tr_velo_to_cam

        self.P2 = np.array(self.data['P2']).reshape(3,4)
        return self

    def leftcam2lidar(self, pts):
        '''
        transform the pts from the left camera frame to lidar frame
        pts_lidar  = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pts_lidar_T = inv(self.Tr_velo_to_cam) @ inv(self.R0_rect) @ pts_hT # (4, #pts)
        pts_lidar = pts_lidar_T.T # (#pts, 4)
        return pts_lidar[:, :3]

    def lidar2leftcam(self, pts):
        '''
        transform the pts from the lidar frame to the left camera frame
        pts_cam = R0_rect @ Tr_velo_to_cam @ pts_lidar
        inputs:
            pts(np.array): [#pts, 3]
                points in the lidar frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pts_cam_T = self.R0_rect @ self.Tr_velo_to_cam @ pts_hT # (4, #pts)
        pts_cam = pts_cam_T.T # (#pts, 4)
        return pts_cam[:, :3]

    def leftcam2imgplane(self, pts):
        '''
        project the pts from the left camera frame to left camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
            points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pixels_T = self.P2 @ pts_hT #(3, #pts)
        pixels = pixels_T.T
        pixels[:, 0] /= pixels[:, 2] + 1e-6
        pixels[:, 1] /= pixels[:, 2] + 1e-6
        return pixels[:, :2]

class KittiLabel:
    '''
    class storing KITTI 3d object detection label
        self.data ([KittiObj])
    '''
    def __init__(self, label_path=None):
        self.path = label_path
        self.data = None
        self._objs_box = None
        self._objs_array = None
        self._objs_name = None
        self._objs_score = None
        self._current_frame = None

    def read_label_file(self, no_dontcare=True):
        '''
        read KITTI label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(KittiObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        num_obj = len(self.data)
        self._objs_array = np.zeros((num_obj, 14)).astype(np.float32)
        self._objs_name = []
        self._objs_score = []
        for i, obj in enumerate(self.data):
            # trun, occ, alpha,
            # bbox_l, bbox_t, bbox_r, bbox_b,
            # h, w, l, x, y, z, ry
            self._objs_array[i, :] = np.array([obj.truncated, obj.occluded, obj.alpha,\
                                               obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b, \
                                               obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry])
            self._objs_name.append(obj.type)
            self._objs_score.append(obj.score)
        self._current_frame = Frame.Cam2
        return self

    @property
    def bboxes3d(self):
        return self._objs_array[:, -7:]

    @property
    def bboxes2d_cam(self):
        return self._objs_array[:, 3:7]

    @property
    def bboxes_score(self):
        return self._objs_score

    @property
    def have_score(self):
        return not None in self._objs_score

    @property
    def have_dontcare(self):
        return "DontCare" in self._objs_name

    @property
    def bboxes_name(self):
        return self._objs_name

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, frame: [Frame, str]):
        if isinstance(frame, str):
            self._current_frame = Frame[frame]
        elif isinstance(frame, Frame):
            self._current_frame = frame
        else:
            print(type(frame))
            raise NotImplementedError

    def add_obj(self, obj):
        if len(self) == 0:
            self.data = []
            self._objs_name = []
            self._objs_score = []
        self.data.append(obj)
        self._objs_name.append(obj.type)
        self._objs_score.append(obj.score)
        tmp = np.array([obj.truncated, obj.occluded, obj.alpha,
                        obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b,
                        obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry]).reshape(1, -1)
        self._objs_array = (np.concatenate([self._objs_array, tmp], axis=0)
                            if self._objs_array is not None else tmp)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __str__(self):
        '''
        TODO: Unit TEST
        '''
        s = ''
        if self.data is not None:
            for obj in self.data:
                s += obj.__str__() + '\n'
        else:
            pass
        return s

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for KittiLabel
        inputs:
            label: KittiLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        if len(self.data) != len(label.data):
            return False
        if len(self.data) == 0:
            return True
        bool_list = []
        for obj1 in self.data:
            bool_obj1 = False
            for obj2 in label.data:
                bool_obj1 = bool_obj1 or obj1.equal(obj2, acc_cls, rtol)
            bool_list.append(bool_obj1)
        return any(bool_list)

    def isempty(self):
        '''
        return True if self.data = None or self.data = []
        '''
        return self.data is None or len(self.data) == 0

class KittiObj():
    '''
    class storing a KITTI 3d object
    '''
    def __init__(self, s=None):
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.bbox_l = None
        self.bbox_t = None
        self.bbox_r = None
        self.bbox_b = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.ry = None
        self.score = None
        if s is None:
            return
        if len(s.split()) == 15: # data
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        elif len(s.split()) == 16: # result
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        else:
            raise NotImplementedError

    def __str__(self):
        if self.score is None:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        else:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score)

    def get_pts(self, pc, calib):
        '''
        get points from pc
        inputs:
            pc: (np.array) [#pts, 3]
                point cloud in velodyne frame
            calib (KittiCalib)
        '''
        idx = self.get_pts_idx(pc, calib)
        return pc[idx]

    def get_pts_idx(self, pc, calib):
        '''
        get points from pc
        inputs:
            pc: (np.array) [#pts, 3]
                point cloud in velodyne frame
            calib (KittiCalib)
        '''
        bottom_Fcam = np.array([self.x, self.y, self.z]).reshape(1, 3)
        center_Fcam = bottom_Fcam + np.array([0, -self.h/2.0, 0]).reshape(1, 3)
        center_Flidar = calib.leftcam2lidar(center_Fcam)
        pc_ = utils.apply_tr(pc, -center_Flidar)
        pc_ = utils.apply_R(pc_, utils.rotz(self.ry+np.pi/2))
        idx_x = np.logical_and(pc_[:, 0] <= self.l/2.0, pc_[:, 0] >= -self.l/2.0)
        idx_y = np.logical_and(pc_[:, 1] <= self.w/2.0, pc_[:, 1] >= -self.w/2.0)
        idx_z = np.logical_and(pc_[:, 2] <= self.h/2.0, pc_[:, 2] >= -self.h/2.0)
        idx = np.logical_and(idx_x, np.logical_and(idx_y, idx_z))
        return idx

    def get_bbox3dcorners(self):
        '''
        get the 8 corners of the bbox3d in camera frame.
        1.--.2
         |  |
         |  |
        4.--.3 (bottom)

        5.--.6
         |  |
         |  |
        8.--.7 (top)

        Camera Frame:
                   ^z
                   |
                y (x)----->x
        '''
        # lwh <-> xzy
        l, w, h = self.l, self.w, self.h
        x, z, y = self.x, self.z, self.y
        bottom = np.array([
            [-l/2, 0, w/2],
            [l/2, 0, w/2],
            [l/2, 0, -w/2],
            [-l/2, 0, -w/2],
        ])
        bottom = utils.apply_R(bottom, utils.roty(self.ry))
        bottom = utils.apply_tr(bottom, np.array([x, y, z]).reshape(-1, 3))
        top = utils.apply_tr(bottom, np.array([0, -h, 0]))
        return np.vstack([bottom, top])
    
    def from_corners(self, calib, corners, cls, score):
        '''
        initialize from corner points
        inputs:
            corners (np.array) [8,3]
                corners in camera frame
                orders
            [-l/2, 0,  w/2],
            [ l/2, 0,  w/2],
            [ l/2, 0, -w/2],
            [-l/2, 0, -w/2],
            [-l/2, h,  w/2],
            [ l/2, h,  w/2],
            [ l/2, h, -w/2],
            [-l/2, h, -w/2],
            cls (str): 'Car', 'Pedestrian', 'Cyclist'
            score (float): 0-1
        '''
        # assert cls in ["Car", "Pedestrian", "Cyclist", "Van",
        #                 "Tram", "Truck", "Person_sitting"]
        assert score <= 1.0
        assert score >= 0.0
        self.x = np.sum(corners[:, 0], axis=0)/ 8.0
        self.y = np.sum(corners[0:4, 1], axis=0)/ 4.0
        self.z = np.sum(corners[:, 2], axis=0)/ 8.0
        self.h = np.sum(abs(corners[4:, 1] - corners[:4, 1])) / 4.0
        self.l = np.sum(
            np.sqrt(np.sum((corners[0, [0, 2]] - corners[1, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[2, [0, 2]] - corners[3, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[4, [0, 2]] - corners[5, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[6, [0, 2]] - corners[7, [0, 2]])**2))
            ) / 4.0
        self.w = np.sum(
            np.sqrt(np.sum((corners[0, [0, 2]] - corners[3, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[1, [0, 2]] - corners[2, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[4, [0, 2]] - corners[7, [0, 2]])**2)) +
            np.sqrt(np.sum((corners[5, [0, 2]] - corners[6, [0, 2]])**2))
            ) / 4.0
        self.ry = np.sum(
            math.atan2(corners[2, 0] - corners[1, 0], corners[2, 2] - corners[1, 2]) +
            math.atan2(corners[6, 0] - corners[5, 0], corners[6, 2] - corners[5, 2]) +
            math.atan2(corners[3, 0] - corners[0, 0], corners[3, 2] - corners[0, 2]) +
            math.atan2(corners[7, 0] - corners[4, 0], corners[7, 2] - corners[4, 2]) +
            math.atan2(corners[0, 2] - corners[1, 2], corners[1, 0] - corners[0, 0]) +
            math.atan2(corners[4, 2] - corners[5, 2], corners[5, 0] - corners[4, 0]) +
            math.atan2(corners[3, 2] - corners[2, 2], corners[2, 0] - corners[3, 0]) +
            math.atan2(corners[7, 2] - corners[6, 2], corners[6, 0] - corners[7, 0])
        ) / 8.0 + np.pi  / 2.0
        if np.isclose(self.ry, np.pi/2.0):
            self.ry = 0.0
        cns_Fcam2d = calib.leftcam2imgplane(corners)
        minx = int(np.min(cns_Fcam2d[:, 0]))
        maxx = int(np.max(cns_Fcam2d[:, 0]))
        miny = int(np.min(cns_Fcam2d[:, 1]))
        maxy = int(np.max(cns_Fcam2d[:, 1]))
        self.ry = utils.clip_ry(self.ry)
        self.type = cls
        self.score = score
        self.truncated = 0
        self.occluded = 0
        self.alpha = 0
        self.bbox_l = minx
        self.bbox_t = miny
        self.bbox_r = maxx
        self.bbox_b = maxy
        return self

    def equal(self, obj, acc_cls, rtol):
        '''
        equal oprator for KittiObj
        inputs:
            obj: KittiObj
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Note: For ry, return True if obj1.ry == obj2.ry + n * pi
        '''
        assert isinstance(obj, KittiObj)
        return (self.type in acc_cls and
                obj.type in acc_cls and
                np.isclose(self.h, obj.h, rtol) and
                np.isclose(self.l, obj.l, rtol) and
                np.isclose(self.w, obj.w, rtol) and
                np.isclose(self.x, obj.x, rtol) and
                np.isclose(self.y, obj.y, rtol) and
                np.isclose(self.z, obj.z, rtol) and
                np.isclose(math.cos(2 * (self.ry - obj.ry)), 1, rtol))

    def copy(self):
        import copy
        return copy.deepcopy(self)

class KittiData:
    '''
    class storing a frame of KITTI data
    '''
    def __init__(self, root_dir, idx, output_dict=None):
        '''
        inputs:
            root_dir(str): kitti dataset dir
            idx(str %6d): data index e.g. "000000"
        '''
        self.calib_path = os.path.join(root_dir, "calib", idx+'.txt')
        self.image2_path = os.path.join(root_dir, "image_2", idx+'.png')
        self.label2_path = os.path.join(root_dir, "label_2", idx+'.txt')
        self.velodyne_path = os.path.join(root_dir, "velodyne", idx+'.bin')
        self.output_dict = output_dict
        if self.output_dict is None:
            self.output_dict = {
                "calib": True,
                "image": True,
                "label": True,
                "velodyne": True
            }

    def read_data(self, num_feature=4):
        '''
        read data
        returns:
            calib(KittiCalib)
            image(np.array): [w, h, 3]
            label(KittiLabel)
            pc(np.array): [# of points, 4]
                point cloud in lidar frame.
                [x, y, z]
                      ^x
                      |
                y<----.z
        '''
        calib = KittiCalib(self.calib_path).read_calib_file() if self.output_dict["calib"] else None
        image = utils.read_image(self.image2_path) if self.output_dict["image"] else None
        label = KittiLabel(self.label2_path).read_label_file() if self.output_dict["label"] else None
        pc = utils.read_pc_from_bin(self.velodyne_path, num_feature=num_feature) if self.output_dict["velodyne"] else None
        return calib, image, label, pc

if __name__ == "__main__":
    label = KittiLabel("/usr/app/data/KITTI/training/label_2/000009.txt").read_label_file()
