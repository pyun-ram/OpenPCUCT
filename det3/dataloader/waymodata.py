import numpy as np
import math
import os
from numpy.linalg import inv
from enum import Enum
from det3.dataloader.carladata import CarlaCalib, CarlaObj, CarlaLabel, CarlaData
try:
    from ..utils import utils
except:
    import det3.utils.utils as utils

Frame = Enum('Frame', ('IMU'))

class WaymoCalib(CarlaCalib):
    pass

class WaymoObj(CarlaObj):
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
            cls (str): 'Car', 'Pedestrian', 'Cyclist', 'Sign'
            score (float): 0-1
        '''
        assert cls in ['Car', 'Pedestrian', 'Cyclist', 'Sign'], cls
        assert score <= 1.0
        assert score >= 0.0
        x_Fcam = np.sum(corners[:, 0], axis=0)/ 8.0
        y_Fcam = np.sum(corners[0:4, 1], axis=0)/ 4.0
        z_Fcam = np.sum(corners[:, 2], axis=0)/ 8.0
        xyz_FIMU = calib.cam2imu(np.array([x_Fcam, y_Fcam, z_Fcam]).reshape(1,3))
        self.x, self.y, self.z = xyz_FIMU.reshape(3)
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
            math.atan2(corners[2, 2] - corners[1, 2], corners[2, 0] - corners[1, 0]) +
            math.atan2(corners[6, 2] - corners[5, 2], corners[6, 0] - corners[5, 0]) +
            math.atan2(corners[3, 2] - corners[0, 2], corners[3, 0] - corners[0, 0]) +
            math.atan2(corners[7, 2] - corners[4, 2], corners[7, 0] - corners[4, 0]) +
            math.atan2(corners[1, 0] - corners[0, 0], corners[0, 2] - corners[1, 2]) +
            math.atan2(corners[5, 0] - corners[4, 0], corners[4, 2] - corners[5, 2]) +
            math.atan2(corners[2, 0] - corners[3, 0], corners[3, 2] - corners[2, 2]) +
            math.atan2(corners[6, 0] - corners[7, 0], corners[7, 2] - corners[6, 2])
        ) / 8.0 + np.pi  / 2.0
        if np.isclose(self.ry, np.pi/2.0):
            self.ry = 0.0
        cns_Fcam2d = calib.cam2imgplane(corners)
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

class WaymoLabel(CarlaLabel):
    def read_label_file(self, no_dontcare=True):
        '''
        read Waymo label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(WaymoObj(s))
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
        self._current_frame = Frame.IMU
        return self

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

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for WaymoLabel
        inputs:
            label: WaymoLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        assert utils.istype(label, "WaymoLabel")
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

class WaymoData(CarlaData):
    def read_data(self):
        '''
        read data
        returns:
            calib(CarlaCalib)
            label(CarlaLabel)
            pc(dict):
                point cloud in Lidar <tag> frame.
                pc[tag] = np.array [#pts, 3], tag is the name of the dir saving velodynes
        '''
        calib = WaymoCalib(self.calib_path).read_calib_file() if self.output_dict["calib"] else None
        label = WaymoLabel(self.label_path).read_label_file() if self.output_dict["label"] else None
        if self.output_dict["velodyne"]:
            pc = dict()
            for k, v in zip(self.velodyne_list, self.velodyne_paths):
                assert k == v.split('/')[-2]
                pc[k] = utils.read_pc_from_npy(v)
        else:
            pc = None
        return pc, label, calib

if __name__ == "__main__":
    from det3.visualizer.vis import BEVImage, FVImage
    from PIL import Image
    import os
    os.makedirs('/usr/app/vis/dev/bev/', exist_ok=True)
    os.makedirs('/usr/app/vis/dev/fv/', exist_ok=True)
    for i in range(0, 5):
        tag = "{:06d}".format(i)
        pc, label, calib = WaymoData('/usr/app/dataset/waymo/data_list_train/', tag).read_data()
        bevimg =  BEVImage(x_range=(-50, 50), y_range=(-50, 50), grid_size=(0.05, 0.05))
        point_cloud = np.vstack([calib.lidar2imu(pc['velo_top'], key='Tr_imu_to_velo_top'),
                                 calib.lidar2imu(pc['velo_front'], key='Tr_imu_to_velo_front'),
                                 calib.lidar2imu(pc['velo_side_left'], key='Tr_imu_to_velo_side_left'),
                                 calib.lidar2imu(pc['velo_side_right'], key='Tr_imu_to_velo_side_right'),
                                 calib.lidar2imu(pc['velo_rear'], key='Tr_imu_to_velo_rear'),
                                ])
        bevimg.from_lidar(point_cloud)
        for obj in label.data:
            bevimg.draw_box(obj, calib)
            print(obj)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save("/usr/app/vis/dev/bev/{}.png".format(tag))
        fvimg = FVImage()
        fvimg.from_lidar(calib, calib.lidar2imu(pc['velo_top'], key='Tr_imu_to_velo_top'))
        for obj in label.data:
            fvimg.draw_box(obj, calib)
            print(obj)
        fvimg_img = Image.fromarray(fvimg.data)
        fvimg_img.save('/usr/app/vis/dev/fv/{}.png'.format(tag))


#
