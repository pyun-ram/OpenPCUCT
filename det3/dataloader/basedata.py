'''
 File Created: Thu Mar 19 2020

 Note: This file contains the ABC for implementing Dataloader of 3D object detection datasets.
'''
from abc import ABC, abstractmethod
from enum import Enum
import math
import copy
import numpy as np
from det3.utils import utils
from det3.ops import crop_pts_3drot, get_corner_box_3drot

class BaseFrame(ABC):
    """
    This ABC is to define the frame.
    """
    Frame = Enum('Frame', ('Ego'))
    def __init__(self, value=None):
        if isinstance(value, str):
            self._frame = self.Frame[value.upper()]
        elif isinstance(value, Enum):
            self._frame = value
        elif value is None:
            self._frame = None
        else:
            raise TypeError("the type of value should be string or enum.")

    @property
    def frame(self):
        assert self._frame is not None
        return self._frame
    
    @frame.setter
    def frame(self, value):
        '''
        @value: str/Enum
        '''
        if isinstance(value, str):
            self._frame = self.Frame[value.upper()]
        elif isinstance(value, Enum):
            self._frame = value
        else:
            raise TypeError("the type of value should be string or enum.")

    @classmethod
    @abstractmethod
    def all_frames():
        '''
        ->return a string list all available frames
        '''
        raise NotImplementedError

    def __hash__(self):
        return hash(self.frame.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.frame == self.Frame[other.upper()]
        elif isinstance(other, BaseFrame):
            return self.frame == other.frame
        elif isinstance(other, Enum):
            return self.frame == other
        else:
            raise TypeError("the type of other should be string or enum.")

    def __repr__(self):
        return self.frame.name

class BaseCalib(ABC):
    """
    This ABC is to define the calib class helping
    coordinate transformation between frames.
    """
    def __init__(self, path):
        self._path = path
        self._data = None

    @abstractmethod
    def read_calib_file(self):
        '''
        Read the self._path to get calib information and write into self._data
        '''
        raise NotImplementedError

class BaseLabel(ABC):
    """
    This ABC is to define the label class helping manage the label.
    """
    def __init__(self, path):
        self._path = path
        self._objs = []
        self._objs_boxes = None
        self._objs_classes = []
        self._objs_scores = []
        self._current_frame = None
        self._cnter = 0

    @abstractmethod
    def read_label_file(self):
        '''
        Read the label file and write into self._objs*.
        You have to set the self._current_frame.
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def box_order(self):
        '''
        ->return a string specifing the order of self._objs_boxes.
        e.g. "x, y, z, h, w, l, ry"
        '''
        raise NotImplementedError

    @property
    def boxes(self):
        return self._objs_boxes
    
    @property
    def scores(self):
        return self._objs_scores
    
    @property
    def classes(self):
        return self._objs_classes
    
    @property
    def current_frame(self):
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value
    
    def add_obj(self, obj):
        '''
        @obj: a derived class of BaseObj
        '''
        self._objs.append(obj)
        self._objs_classes.append(obj.cls)
        self._objs_scores.append(obj.score)
        self._objs_boxes = (np.concatenate([self._objs_boxes, obj.array], axis=0)
                            if self._objs_boxes is not None else obj.array)

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, idx):
        '''
        ->return a derived class of BaseObj
        '''
        return self._objs[idx]

    def equal(self, other, acc_cls=None, atol=1e-2):
        '''
        Return True if it can find a same counterpart in other.
        @other: a derived class of BaseLabel
        @acc_cls: None/List
            The accurate classes count True. e.g. ['Car', 'Van']
        @atol: float
        Note: return True if len(self) == 0
        '''
        if len(self) == 0:
            return True
        bool_list = []
        for obj1 in self:
            bool_obj1 = False
            for obj2 in other:
                bool_obj1 = bool_obj1 or obj1.equal(obj2, acc_cls, atol)
            bool_list.append(bool_obj1)
        return any(bool_list)

    def __len__(self):
        return len(self._objs)

    def __str__(self):
        s = ''
        for obj in self:
            s += str(obj) + '\n'
        return s

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self._cnter >= len(self):
            self._cnter = 0
            raise StopIteration
        else:
            res = self[self._cnter]
            self._cnter += 1
            return res

class BaseObj(ABC):
    """
    This ABC is to define the object helping manage one single object.
    """
    @abstractmethod
    def __init__(self, arr, cls, score):
        '''
        @arr: np.array [7]
        [x, y, z] - bottom center
        [l, w, h] - scale: x- y- z- axis
        theta- rotation along z axis
        '''
        self.x = None
        self.y = None
        self.z = None
        self.l = None
        self.w = None
        self.h = None
        self.theta = None
        self.cls = cls
        self.score = score
        self._current_frame = None
        (self.x, self.y, self.z, 
            self.l, self.w, self.h, self.theta) = arr.flatten()

    @abstractmethod
    def __str__(self):
        '''
        -> return a string representing the object
        '''
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    @property
    def array(self):
        '''
        return an np.ndarray representation
        '''
        return np.array([self.x, self.y, self.z,
                         self.l, self.w, self.h,
                         self.theta]).reshape(1, -1)

    @abstractmethod
    def equal(self, other, acc_cls, atol):
        '''
        Return True if it is same to other.
        @other: a derived class of BaseObj
        @acc_cls: None/List
            The accurate classes count True. e.g. ['Car', 'Van']
        @atol: float
        Note: return True if len(self) == 0
        '''
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    @property
    def current_frame(self):
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value

    def get_pts_idx(self, pc):
        '''
        Get the index of pts in the bounding box.
        @ pc: np.array, torch.Tensor, torch.Tensor.cuda
            The object has to be in a same frame of pc.
        -> return a list of idxes.
        '''
        return crop_pts_3drot(self.array, pc[:, :3])[0]

    def get_bbox3d_corners(self):
        '''
        Get the corners defined by the bounding box
        1.--.2
         |  |                  ^x
         |  |                  |
        4.--.3 (bottom) y<----.z
        5.--.6
         |  |
         |  |
        8.--.7 (top)
        -> return a np.array [8, 3]
        '''
        return get_corner_box_3drot(self.array)[0]

    def from_corners(self, corners, cls=None, score=None):
        '''
        Define the bounding box from the corners
        @ corners: np.ndarray [8, 3] with a same definition of BaseObj.get_bbox3d_corners()
            in UdiFrame("BASE")
        @ cls: str
        @ score: float [0, 1] / None
        '''
        x_Fbase = np.sum(corners[:, 0], axis=0)/ 8.0
        y_Fbase = np.sum(corners[:, 1], axis=0)/ 8.0
        z_Fbase = np.sum(corners[:4, 2], axis=0)/ 4.0
        self.x, self.y, self.z = x_Fbase, y_Fbase, z_Fbase
        self.h = np.sum(abs(corners[4:, 2] - corners[:4, 2])) / 4.0
        self.l = np.sum(
            np.sqrt(np.sum((corners[0, [0, 1]] - corners[1, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[2, [0, 1]] - corners[3, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[4, [0, 1]] - corners[5, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[6, [0, 1]] - corners[7, [0, 1]])**2))
            ) / 4.0
        self.w = np.sum(
            np.sqrt(np.sum((corners[0, [0, 1]] - corners[3, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[1, [0, 1]] - corners[2, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[4, [0, 1]] - corners[7, [0, 1]])**2)) +
            np.sqrt(np.sum((corners[5, [0, 1]] - corners[6, [0, 1]])**2))
            ) / 4.0
        self.theta = np.sum(
            math.atan2(corners[2, 1] - corners[1, 1], corners[2, 0] - corners[1, 0]) +
            math.atan2(corners[6, 1] - corners[5, 1], corners[6, 0] - corners[5, 0]) +
            math.atan2(corners[3, 1] - corners[0, 1], corners[3, 0] - corners[0, 0]) +
            math.atan2(corners[7, 1] - corners[4, 1], corners[7, 0] - corners[4, 0]) +
            math.atan2(corners[1, 0] - corners[0, 0], corners[0, 1] - corners[1, 1]) +
            math.atan2(corners[5, 0] - corners[4, 0], corners[4, 1] - corners[5, 1]) +
            math.atan2(corners[2, 0] - corners[3, 0], corners[3, 1] - corners[2, 1]) +
            math.atan2(corners[6, 0] - corners[7, 0], corners[7, 1] - corners[6, 1])
        ) / 8.0
        if np.isclose(self.theta, np.pi/2.0):
            self.theta = 0.0
        self.theta = utils.clip_ry(self.theta)
        self.cls = cls
        self.score = score
        return self

class BaseData(ABC):
    """
    This ABC is to define the class helping read a package of a dataset.
    """
    @abstractmethod
    def __init__(self, root_dir, tag):
        '''
        To init the root_dir and idx and other dirs, like LiDAR dirs...
        @ root_dir: str
        @ tag: str
        '''
        self._root_dir = root_dir
        self._tag = tag

    @abstractmethod
    def read_data(self):
        '''
        To read data and return a dict
        -> return dict
        '''
        raise NotImplementedError

