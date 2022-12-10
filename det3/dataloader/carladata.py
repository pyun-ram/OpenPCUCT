'''
File Created: Tuesday, 23rd April 2019 11:23:50 am
'''
import numpy as np
import math
import os
from numpy.linalg import inv
from enum import Enum
try:
    from ..utils import utils
except:
    # Run script python3 dataloader/carladata.py
    import det3.utils.utils as utils

Frame = Enum('Frame', ('IMU'))

class CarlaCalib:
    '''
    class storing CARLA calib data

              ^x
              |
        y<--.(z)        LiDAR Frame and IMU Frame

              ^ z
              |
           y (x)---> x Camera Frame
        The original point of IMU Frame is the same to Camera Frame in world Frame.
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None
        self.num_of_lidar = None
        # Note: Dangerous hard code here:
        # If modify this P0, you have to modify the
        #   tools/evaluate_object_3d_offline_carla.cpp
        #   unit-test/test_dataloader_carladata.py
        self.P0 = np.array([[450, 0., 600, 0.],
                            [0., 450, 180, 0.],
                            [0., 0., 1, 0.]])
    def copy(self):
        import copy
        return copy.deepcopy(self)

    def read_calib_file(self):
        '''
        read CARLA calib file
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
        self.num_of_lidar = sum(['Tr_imu_to_velo' in itm for itm in self.data.keys()])
        return self

    def lidar2imu(self, pts, key):
        '''
        convert pts in lidar(<key> frame) to imu frame
        inputs:
            pts (np.array): [#pts, 3]
                point cloud in lidar<key> frame
            key (str):  'Tr_imu_to_velo_XX',
                        It should be corresopnd to self.data.keys()
        returns:
            pts_imu (np.array): [#pts, 3]
                point cloud in imu frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        assert pts.shape[1] == 3
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        Tr_imu_to_velo = np.array(self.data[key]).reshape(4, 4)
        Tr_velo_to_imu = inv(Tr_imu_to_velo)
        pts_imu_T = Tr_velo_to_imu @ pts_hT # (4, #pts)
        pts_imu = pts_imu_T.T
        return pts_imu[:, :3]

    def imu2cam(self, pts):
        '''
        convert pts in imu frame to cam frame
        inputs:
            pts (np.array): [#pts, 3]
                point clouds in imu frame
        returns:
            pts_cam (np.array): [#pts, 3]
                point cloud in camera frame
        '''
        assert pts.shape[1] == 3
        pts_imu = pts
        pts_imu_x = pts_imu[:, 0:1]
        pts_imu_y = pts_imu[:, 1:2]
        pts_imu_z = pts_imu[:, 2:3]
        return np.hstack([-pts_imu_y, -pts_imu_z, pts_imu_x])

    def cam2imu(self, pts):
        '''
        convert pts in camera frame to imu frame
        inputs:
            pts (np.array): [#pts, 3]
                point clouds in camera frame
        returns:
            pts_imu (np.array): [#pts, 3]
                point cloud in imu frame
        '''
        assert pts.shape[1] == 3
        pts_cam = pts
        pts_cam_x = pts_cam[:, 0:1]
        pts_cam_y = pts_cam[:, 1:2]
        pts_cam_z = pts_cam[:, 2:3]
        return np.hstack([pts_cam_z, -pts_cam_x, -pts_cam_y])

    def cam2imgplane(self, pts):
        '''
        project the pts from the camera frame to camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in camera frame
        return:
            pixels: [#pts, 2]
                pixels on the image
        Note: the returned pixels are floats
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pixels_T = self.P0 @ pts_hT #(3, #pts)
        pixels = pixels_T.T
        pixels[:, 0] /= pixels[:, 2]
        pixels[:, 1] /= pixels[:, 2]
        return pixels[:, :2]

class CarlaObj():
    '''
    class storing a Carla 3d object
    Defined in IMU frame
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

    def get_pts(self, pc, calib=None):
        '''
        get points from pc
        inputs:
            pc: (np.array) [#pts, 3]
                in imu frame
        '''
        idx = self.get_pts_idx(pc, calib)
        return pc[idx]        

    def get_pts_idx(self, pc, calib=None):
        '''
        get points from pc
        inputs:
            pc: (np.array) [#pts, 3]
                in imu frame
        '''
        bottom_Fimu = np.array([self.x, self.y, self.z]).reshape(1, 3)
        center_Fimu = bottom_Fimu + np.array([0, 0, self.h/2.0]).reshape(1, 3)
        pc_ = utils.apply_tr(pc, -center_Fimu)
        pc_ = utils.apply_R(pc_, utils.rotz(-self.ry))
        idx_x = np.logical_and(pc_[:, 0] <= self.l/2.0, pc_[:, 0] >= -self.l/2.0)
        idx_y = np.logical_and(pc_[:, 1] <= self.w/2.0, pc_[:, 1] >= -self.w/2.0)
        idx_z = np.logical_and(pc_[:, 2] <= self.h/2.0, pc_[:, 2] >= -self.h/2.0)
        idx = np.logical_and(idx_x, np.logical_and(idx_y, idx_z))
        return idx

    def get_bbox3dcorners(self):
        '''
        get the 8 corners of the bbox3d in imu frame.
        1.--.2
         |  |
         |  |
        4.--.3 (bottom)

        5.--.6
         |  |
         |  |
        8.--.7 (top)

        IMU Frame:
                      ^x
                      |
                y<----.z
        '''
        # lwh <-> yxz (imu) # lwh <-> xyz
        #  (origin comment) # (new comment)
        l, w, h = self.l, self.w, self.h
        x, z, y = self.x, self.z, self.y
        bottom = np.array([
            [-l/2,  w/2, 0],
            [ l/2,  w/2, 0],
            [ l/2, -w/2, 0],
            [-l/2, -w/2, 0],
        ])
        bottom = utils.apply_R(bottom, utils.rotz(self.ry))
        bottom = utils.apply_tr(bottom, np.array([x, y, z]).reshape(-1, 3))
        top = utils.apply_tr(bottom, np.array([0, 0, h]))
        return np.vstack([bottom, top])

    def equal(self, obj, acc_cls, rtol):
        '''
        equal oprator for CarlaObj
        inputs:
            obj: CarlaObj
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Note: For ry, return True if obj1.ry == obj2.ry + n * pi
        '''
        assert isinstance(obj, CarlaObj)
        return (self.type in acc_cls and
                obj.type in acc_cls and
                np.isclose(self.h, obj.h, rtol) and
                np.isclose(self.l, obj.l, rtol) and
                np.isclose(self.w, obj.w, rtol) and
                np.isclose(self.x, obj.x, rtol) and
                np.isclose(self.y, obj.y, rtol) and
                np.isclose(self.z, obj.z, rtol) and
                np.isclose(math.cos(2 * (self.ry - obj.ry)), 1, rtol))

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
        assert cls in ['Car', 'Pedestrian', 'Cyclist']
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

    def copy(self):
        import copy
        return copy.deepcopy(self)

class CarlaLabel:
    '''
    class storing Carla 3d object detection label
        self.data ([CarlaObj])
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
        read CARLA label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(CarlaObj(s))
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
        for obj in self.data:
            s += obj.__str__() + '\n'
        return s

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for CarlaLabel
        inputs:
            label: CarlaLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        assert utils.istype(label, "CarlaLabel")
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

class CarlaData:
    '''
    class storing a frame of Carla data
    Notes:
        The dir should be:
        calib/
            xx.txt
        label_imu/
            xx.txt
        velo_xx/
            xx.pcd
        velo_xx/
        velo_xx/
    '''
    def __init__(self, root_dir, idx, output_dict=None):
        '''
        inputs:
            root_dir(str): carla dataset dir
            idx(str %6d): data index e.g. "000000"
        '''
        self.calib_path = os.path.join(root_dir, "calib", idx+'.txt')
        self.label_path = os.path.join(root_dir, "label_imu", idx+'.txt')

        velodyne_list = os.listdir(root_dir)
        self.velodyne_list = [itm for itm in velodyne_list if itm.split('_')[0]=="velo"]
        self.velodyne_paths = [os.path.join(root_dir, itm, idx+'.npy') for itm in self.velodyne_list]
        self.output_dict = output_dict
        if self.output_dict is None:
            self.output_dict = {
                "calib": True,
                "label": True,
                "velodyne": True
            }
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
        calib = CarlaCalib(self.calib_path).read_calib_file() if self.output_dict["calib"] else None
        label = CarlaLabel(self.label_path).read_label_file() if self.output_dict["label"] else None
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
    for i in range(0, 300):
        tag = "{:06d}".format(i)
        pc, label, calib = CarlaData('/usr/app/data/CARLA/dev/', tag).read_data()
        bevimg =  BEVImage(x_range=(-50, 50), y_range=(-50, 50), grid_size=(0.05, 0.05))
        point_cloud = np.vstack([calib.lidar2imu(pc['velo_top'], key='Tr_imu_to_velo_top'),
                                 calib.lidar2imu(pc['velo_left'], key='Tr_imu_to_velo_left'),
                                 calib.lidar2imu(pc['velo_right'], key='Tr_imu_to_velo_right'),
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


