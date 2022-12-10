'''
 File Created: Thu Mar 19 2020
'''
from enum import Enum
import os
import math
import numpy as np
from numpy.linalg import inv
from det3.dataloader.basedata import *
from det3.ops import read_txt, apply_T, read_json, hfill_pts, read_npy

class LyftFrame(BaseFrame):
    Frame = Enum('Frame', ('BASE', 'LIDARTOP', 'LIDARLEFT', 'LIDARRIGHT', 'VCAM'))

    def all_frames():
        return 'BASE, LIDARTOP, LIDARFRONT, LIDARLEFT, LIDARRIGHT, VCAM'

class LyftCalib(BaseCalib):
    def __init__(self, path):
        super().__init__(path)
        # intrinsic matrix of virtual camera
        self._vcam_P = np.array([[450, 0., 1200, 0.],
                                [0., 450, 400, 0.],
                                [0., 0., 1, 0.]]).astype(np.float32)
        # extrinsic matrix of virtual camera
        self._vcam_T = np.eye(4).astype(np.float32)

    @property
    def vcam_P(self):
        return self._vcam_P

    @vcam_P.setter
    def vcam_P(self, value):
        self._vcam_P = value

    @property
    def vcam_T(self):
        return self._vcam_T

    @vcam_T.setter
    def vcam_T(self, value):
        self._vcam_T = value
        self._data[(LyftFrame("VCAM"), LyftFrame("BASE"))] = value
        self._data[(LyftFrame("BASE"), LyftFrame("VCAM"))] = inv(value)

    @staticmethod
    def _calib_file_key_to_frame(s):
        map_dict = {
            "Tr_imu_to_velo_top": (LyftFrame("BASE"), LyftFrame("LIDARTOP")),
            "Tr_imu_to_velo_front_left": (LyftFrame("BASE"), LyftFrame("LIDARLEFT")),
            "Tr_imu_to_velo_front_right": (LyftFrame("BASE"), LyftFrame("LIDARRIGHT")),
            "Tr_velo_top_to_base": (LyftFrame("LIDARTOP"), LyftFrame("BASE")),
            "Tr_velo_front_left_to_imu": (LyftFrame("LIDARLEFT"), LyftFrame("BASE")),
            "Tr_velo_front_right_to_imu": (LyftFrame("LIDARRIGHT"), LyftFrame("BASE")),
            "Tr_imu_to_imu": (LyftFrame("BASE"), LyftFrame("BASE"))
        }
        return map_dict[s]

    def read_calib_file(self):
        '''
        '''
        calib = dict()
        str_list = read_txt(self._path)
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            if itm.split(':')[0] == "Tr_imu_to_cam_front":
                continue
            calib[LyftCalib._calib_file_key_to_frame(itm.split(':')[0])] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = np.array([float(itm) for itm in v.split()]).astype(np.float32).reshape(4, 4)
        inv_calib = dict()
        for k, v in calib.items():
            inv_calib[(k[1], k[0])] = inv(v)
        calib[(LyftFrame("BASE"), LyftFrame("BASE"))] = np.eye(4).astype(np.float32)
        calib[(LyftFrame("VCAM"), LyftFrame("BASE"))] = self.vcam_T
        inv_calib[(LyftFrame("BASE"), LyftFrame("VCAM"))] = inv(self.vcam_T)
        calib = {**calib, **inv_calib}
        self._data = calib
        return self
    
    def transform(self, pts, source_frame: LyftFrame, target_frame: LyftFrame):
        '''
        transform pts from source_frame to target_frame
        @pts: np.array [N, 3]
        @source_frame: LyftFrame
        @target_frame: LyftFrame
        -> pts_t: np.array [N, 3]
        '''
        # get T_ego_s
        T_ego_s = self._data[(source_frame, LyftFrame("BASE"))]
        # get T_t_ego
        T_t_ego = self._data[(LyftFrame("BASE"), target_frame)]
        # get T_t_s = T_t_ego X T_ego_s
        T = T_t_ego @ T_ego_s
        return apply_T(pts[:, :3], T)

    def vcam2imgplane(self, pts):
        '''
        project the pts from the camera frame to camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in virtual camera frame
        return:
            pixels: [#pts, 2]
                pixels on the image
        Note: the returned pixels are floats
        '''
        pts_x, pts_y, pts_z = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
        pts_cam = np.hstack([-pts_y, -pts_z, pts_x])
        pts_cam_h = hfill_pts(pts_cam)
        pixels_T = self.vcam_P @ pts_cam_h.T #(3, #pts)
        pixels = pixels_T.T
        pixels[:, 0] /= (pixels[:, 2]+1e-6)
        pixels[:, 1] /= (pixels[:, 2]+1e-6)
        return pixels[:, :2]

    def lidar2imu(self, pts, key):
        s = LyftFrame(f"LIDAR{key.split('_')[-1].upper()}")
        return self.transform(pts, source_frame=s ,target_frame=LyftFrame("BASE"))

class LyftObj(BaseObj):
    def __init__(self, arr=np.zeros(7), cls=None, score=None, frame="BASE"):
        super().__init__(arr, cls, score)
        self.current_frame = frame
    
    def __str__(self):
        score = "" if self.score is None else f" {self.score:.2f}"
        return f"{self.cls} {self.x:.2f} {self.y:.2f} {self.z:.2f} {self.l:.2f} {self.w:.2f} {self.h:.2f} {self.theta:.2f}{score}"

    def equal(self, other, acc_cls=None, atol=1e-2):
        acc_cls = [other.cls] if acc_cls is None else acc_cls
        return (self.current_frame == other.current_frame and
                self.cls in acc_cls and
                other.cls in acc_cls and
                np.isclose(self.h, other.h, atol) and
                np.isclose(self.l, other.l, atol) and
                np.isclose(self.w, other.w, atol) and
                np.isclose(self.x, other.x, atol) and
                np.isclose(self.y, other.y, atol) and
                np.isclose(self.z, other.z, atol) and
                np.isclose(math.cos(2 * (self.theta - other.theta)), 1, atol))

class LyftLabel(BaseLabel):
    def __init__(self, path=None):
        super().__init__(path)

    def read_label_file(self):
        with open(self._path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            from det3.dataloader.carladata import CarlaObj
            cobj = CarlaObj(s)
            x = cobj.x
            y = cobj.y
            z = cobj.z
            l = cobj.l
            w = cobj.w
            h = cobj.h
            theta = cobj.ry
            arr = np.array([x, y, z, l, w, h, theta])
            cls = cobj.type
            obj = LyftObj(arr, cls, score=None, frame="BASE")
            self.add_obj(obj)
        self.current_frame = LyftFrame("BASE")
        return self

    def box_order(self):
        return "x, y, z, l, w, h, theta"

class LyftData(BaseData):
    def __init__(self, root_dir, tag, output_dict=None):
        super().__init__(root_dir, tag)
        self._calib_path = os.path.join(root_dir, "calib", tag+'.txt')
        self._label_path = os.path.join(root_dir, "label_imu", tag+'.txt')
        self._lidar_list = ["velo_top", "velo_front_left", "velo_front_right"]
        self._lidar_paths = {itm: os.path.join(root_dir, itm, tag+'.npy') for itm in self._lidar_list}
        self._output_dict = output_dict
        if output_dict is None:
            self._output_dict = {
                "calib": True,
                "label": True,
                "lidar": True
            }

    def read_data(self):
        '''
        -> res: dict
            res[calib]: LyftCalib / None
            res[label]: LyftLabel / None
            res[pc]: dict {lidar: np.ndarray (#pts, >=3) (in Flidar)} / None
        '''
        calib = LyftCalib(self._calib_path).read_calib_file() if self._output_dict["calib"] else None
        label = LyftLabel(self._label_path).read_label_file() if self._output_dict["label"] else None
        if self._output_dict["lidar"]:
            pc = dict()
            for k, v in self._lidar_paths.items():
                pc[k] = read_npy(v).reshape(-1, 4)
        else:
            pc = None
        res = {k: None for k in self._output_dict.keys()}
        res["calib"] = calib
        res["label"] = label
        res["lidar"] = pc
        return res

    @staticmethod
    def lidar_to_frame(lidar):
        return ("".join(lidar.split("_"))).upper()
