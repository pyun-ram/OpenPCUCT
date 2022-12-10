'''
Adapted from
https://github.com/ZiningWang/Inferring-Spatial-Uncertainty-in-Object-Detection
'''
import numpy as np
import math
from typing import *
from scipy.stats import multivariate_normal
from pcuct.ops.jiou import jiou_utils

# z
# z_coord
# z_surface is 1D in BEV, from 0 to 4 ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2), l along 1st axis
# z_embed is 2D in BEV
# z_normed = z_embed / [l,w]
z_normed_corners_default = np.array(
	[[1,-1],[1,1],[-1,1],[-1,-1]])/2

def create_BEV_box_sample_grid(sample_grid):
    '''
    Args:
        sample_grid (float): resolution of sample grid (in z space)
    Returns:
        z_normed (np.array (N, N, 2))
    TODO: call it with cpp implementation
    '''
    x = np.arange(-0.5 + sample_grid / 2, 0.5, sample_grid)
    y = x
    zx, zy = np.meshgrid(x, y)
    z_normed = np.concatenate((zx.reshape((-1, 1)), zy.reshape((-1, 1))), axis=1)
    return z_normed

def create_3D_box_sample_grid(sample_grid):
    '''
    Args:
        sample_grid (float): resolution of sample grid (in z space)
    Returns:
        z_normed (np.array (N, N, 3))
    TODO: call it with cpp implementation
    '''
    x = np.arange(-0.5 + sample_grid / 2, 0.5, sample_grid)
    y = x
    z = x
    zx, zy, zz = np.meshgrid(x, y, z)
    z_normed = np.concatenate((
        zx.reshape((-1, 1)),
        zy.reshape((-1, 1)),
        zz.reshape((-1, 1))), axis=1)
    return z_normed

class uncertain_label_BEV:
    def __init__(self,
        boxBEV: List,
        x_std: List=[0.44, 0.10, 0.25, 0.25, 0.1745]):
        '''
        Predictive distribution as a Normal distribution.
        Args:
            boxBEV: mean of the label
                [x, y, length(scale_x), width(scale_y), rot]
            x_std: std of the label
                [x, y, length(scale_x), width(scale_y), rot]
        '''
        self.x0 = np.array(boxBEV).reshape(5)
        ry = self.x0[4]
        self.rotmat = np.array([
            [math.cos(ry), -math.sin(ry)],
            [math.sin(ry), math.cos(ry)]])
        self.feature0 = np.array([
            self.x0[0],
            self.x0[1],
            self.x0[2] * math.cos(ry),
            self.x0[2] * math.sin(ry),
            self.x0[3] * math.cos(ry),
            self.x0[3] * math.sin(ry)])
        self.pos_std = [x_std[0], x_std[1]]  # assume to be the same as dimension?
        self.dim_std = [x_std[2], x_std[3]]  # l,w
        self.ry_std = [x_std[4]]  # rotation, about 10deg.
        # Nobody: precision matrix of feature0
        self.Q0_feature = np.diag(1 / np.array([
        self.pos_std[0],
        self.pos_std[1],
        self.dim_std[0]*abs(math.cos(ry))+self.x0[2]*self.ry_std[0]*abs(math.sin(ry)),
        self.dim_std[0]*abs(math.sin(ry))+self.x0[2]*self.ry_std[0]*abs(math.cos(ry)),
        self.dim_std[1]*abs(math.cos(ry))+self.x0[3]*self.ry_std[0]*abs(math.sin(ry)),
        self.dim_std[1]*abs(math.sin(ry))+self.x0[3]*self.ry_std[0]*abs(math.cos(ry))
        ])**2)
        # Nobody: precision matrix of X
        self.Q0_X = np.diag(1 / np.array([
        self.pos_std[0],
        self.pos_std[1],
        self.dim_std[0],
        self.dim_std[1],
        self.ry_std[0]]) ** 2)
        self.posterior = None
        self.mean_LIDAR_variance = None

    @staticmethod
    def feature0_to_x0(feature0:np.ndarray)->np.ndarray:
        '''
        Convert feature0(6dim) to x0(5dim).
        Args:
            feature0 (6,): [xc, yc, lcosry, lsinry, wcosry, wsinry]
        Return:
            x0 (5,): [xc, yc, l, w, ry]
        '''
        feature0 = feature0.reshape(-1)
        if feature0.shape[0] == 5:
            return feature0
        xc, yc, lcosry, lsinry, wcosry, wsinry = feature0[:]
        l = (lcosry **2 + lsinry **2) **0.5
        w = (wcosry **2 + wsinry **2) **0.5
        tanry1 = lsinry/lcosry
        ry1 = np.arctan(tanry1)
        tanry2 = wsinry/wcosry
        ry2 = np.arctan(tanry2)
        return np.array([xc, yc, l, w, (ry1 + ry2) * 0.5])

    def set_posterior(self, mean, cov):
        '''
        Args:
            mean (np.array (1,6))
            cov (np.array (6,6))
        '''
        self.posterior = (mean, cov)

    def get_corner_norms(self):
        '''
        Returns:
            np.array([
            [0.5,-0.5],
            [0.5,0.5],
            [-0.5,0.5],
            [-0.5,-0.5]
            ])
        '''
        corner_norms = z_normed_corners_default
        return corner_norms

    def embedding(self, x0, z_surface):
        '''
        Embed z_surface according to x0
        Args:
            x0: label [xc, yc, l, w, ry]
            z_surface (np.array (N)): z_surface is 1D in BEV,
                from 0 to 4 ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2),
                l along 1st axis
        Returns:
            z_embed (np.array (N, 2)): 2D points in the object frame (no rotation).
        '''
        assert(x0.shape[0]==5)
        z_surface[z_surface > 4] -= 4
        z_surface[z_surface < 0] += 4
        z_surface.reshape(-1)
        z_embed = np.zeros((z_surface.shape[0], 2))
        temp_idx = np.squeeze(z_surface >= 3)
        z_embed[temp_idx, 1] = ((-(z_surface[temp_idx] - 3) + 0.5) * x0[3]).reshape(-1)
        z_embed[temp_idx, 0] = -x0[2] / 2
        temp_idx = np.squeeze(np.logical_and(z_surface >= 2, z_surface < 3))
        z_embed[temp_idx, 1] = x0[3] / 2
        z_embed[temp_idx, 0] = ((-(z_surface[temp_idx] - 2) + 0.5) * x0[2]).reshape(-1)
        temp_idx = np.squeeze(np.logical_and(z_surface >= 1, z_surface < 2))
        z_embed[temp_idx, 1] = (((z_surface[temp_idx] - 1) - 0.5) * x0[3]).reshape(-1)
        z_embed[temp_idx, 0] = x0[2] / 2
        temp_idx = np.squeeze(np.logical_and(z_surface >= 0, z_surface < 1))
        z_embed[temp_idx, 1] = -x0[3] / 2
        z_embed[temp_idx, 0] = ((z_surface[temp_idx] - 0.5) * x0[2]).reshape(-1)
        return z_embed

    def project(self, x0, z_embed):
        '''
        Project z_embed to z_surface according to x0
        Args:
            x0: label [xc, yc, l, w, ry]
            z_embed (np.array (N, 2)): 2D points in the object frame (no rotation).
        Returns:
            z_surface (np.array (N)): z_surface is 1D in BEV,
                from 0 to 4 ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2),
                l along 1st axis
        '''
        assert(x0.shape[0]==5)
        z_normed = z_embed / x0[2:4]
        amax = np.argmax(np.abs(z_normed), axis=1)
        z_surface = np.zeros((z_normed.shape[0]))

        temp_idx = np.squeeze(np.logical_and(amax == 1, z_normed[:, 1] < 0))
        z_surface[temp_idx] = z_normed[temp_idx, 0] + 0.5
        temp_idx = np.squeeze(np.logical_and(amax == 0, z_normed[:, 0] >= 0))
        z_surface[temp_idx] = z_normed[temp_idx, 1] + 0.5 + 1
        temp_idx = np.squeeze(np.logical_and(amax == 1, z_normed[:, 1] >= 0))
        z_surface[temp_idx] = -z_normed[temp_idx, 0] + 0.5 + 2
        temp_idx = np.squeeze(np.logical_and(amax == 0, z_normed[:, 0] < 0))
        z_surface[temp_idx] = -z_normed[temp_idx, 1] + 0.5 + 3
        return z_surface

    def add_surface_coord(self, z_surface, z_coord):
        return z_surface+z_coord

    def Jacobian_surface(self, z_surface):
        '''
        Compute Jacobian matrix 
        \frac{\partial [x, y]}
        {\partial{xc1, xc2, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w}}
        Args:
            z_surface (np.ndarray (N,))
        Returns:
            J (np.ndarry (N, 2, 6))
        '''
        z_normed = self.embedding(self.x0, z_surface) / self.x0[2:4]
        Jacobian = self.Jacobian_z_normed(z_normed)
        return Jacobian

    def Jacobian_z_normed_x0(self, z_normed):
        '''
        Compute Jacobian matrix 
        \frac{\partial [x, y]}
        {\partial{xc, yc, l, w, ry}}
        Args:
            z_normed (np.ndarray (N, 2))
        Returns:
            J (np.ndarry (N, 2, 5))
        Notes:
        The relationship is:
        [x] = [cosry*l -sinry*w] [z_normed_0] + [x_c]
        [y]   [sinry*l  cosry*w] [z_normed_1]   [y_c]
        The Jacobian J_xy_x0 is:
        \begin{bmatrix}
        \frac{\partial x}{\partial x_c} & \frac{\partial x}{\partial y_c} & \frac{\partial x}{\partial l} & \frac{\partial x}{\partial w} & \frac{\partial x}{\partial \theta}\\ 
        \frac{\partial y}{\partial x_c} & \frac{\partial y}{\partial y_c} & \frac{\partial y}{\partial l} & \frac{\partial y}{\partial w} & \frac{\partial y}{\partial \theta} \end{bmatrix}
        '''
        return jiou_utils.jacobian_z_normed_x0_bev(self.x0, z_normed)

    def Jacobian_z_normed(self, z_normed):
        '''
        Compute Jacobian matrix 
        \frac{\partial [x, y]}
        {\partial{xc1, xc2, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w}}
        Args:
            z_normed (np.ndarray (N, 2))
        Returns:
            J (np.ndarry (N, 2, 6))
        Notes:
        The relationship is:
        [x] = [cosry*l -sinry*w] [z_normed_0] + [x_c]
        [y]   [sinry*l  cosry*w] [z_normed_1]   [y_c]
        The Jacobian J_xy_feature0 is:
        \begin{bmatrix}
        \frac{\partial x}{\partial x_c} & \frac{\partial x}{\partial y_c} & \frac{\partial x}{\partial lcos\theta} & \frac{\partial x}{\partial lsin\theta} & \frac{\partial x}{\partial wcos\theta} & \frac{\partial x}{\partial wsin\theta}\\ 
        \frac{\partial y}{\partial x_c} & \frac{\partial y}{\partial y_c} & \frac{\partial y}{\partial lcos\theta} & \frac{\partial y}{\partial lsin\theta} & \frac{\partial y}{\partial wcos\theta} & \frac{\partial y}{\partial wsin\theta}\end{bmatrix}
        '''
        return jiou_utils.jacobian_z_normed_feat0_bev(z_normed)

    def sample_boundary_points(self):
        '''
        Returns:
            z_surface (np.ndarray (40))
            z_embed (np.ndarray (40, 2))
            z_normed (np.ndarray (40, 2))
        '''
        z_surface = np.arange(0, 4, 0.05)
        z_embed = self.embedding(self.x0, z_surface)
        z_normed = z_embed / self.x0[2:4]
        return z_surface, z_embed, z_normed

    def calc_uncertainty_points(self, z_normed):
        '''
        Propagate the uncertainty from label posterior distribution to 2D points.
        Args:
            z_normed (np.ndarray (N, 2))
        Returns:
            mean (np.ndarray (N, 2))
            cov (np.ndarray (N, 2, 2))
        '''
        return jiou_utils.calc_uncertainty_points_bev(z_normed,
            mean=self.x0, cov=self.posterior[1])

    def sample_prob(self, points, sample_grid=0.1):
        '''
        Sample probability given points.
        Args:
            points (np.ndarray (N, 2)): 2D coordinates
            sample_grid (float): resolution of the sample grid.
        Returns:
            probs (np.ndarray (N)): probability of points in the posterior distribution.
        '''
        return jiou_utils.sample_prob_bev(points,
            mean=self.x0, cov=self.posterior[1],
            sample_grid=sample_grid)

class uncertain_label_3D:
    def __init__(self,
        box3D,
        x_std=[0.44, 0.10, 0.01, 0.25, 0.25, 0.01, 0.1745]):
        '''
        Predictive distribution as a Normal distribution.
        Args:
            box3D: mean of the label
                Nobody: [xc, yc, zc, l(scale_x), w(scale_y), h(scale_z), ry]
            x_std: std of the label
                Nobody: [xc, yc, zc, l(scale_x), w(scale_y), h(scale_z), ry]
        '''
        self.x0 = np.array(box3D).reshape(7)
        ry = self.x0[-1]
        self.rotmat = np.array(
            [[math.cos(ry), -math.sin(ry), 0,],
             [math.sin(ry),  math.cos(ry), 0,],
             [0           ,             0, 1,]])
        # feature0: xc,yc,zc,lcosry,lsinry,wcosry,wsinry,h
        self.feature0 = np.array(
            [self.x0[0],
             self.x0[1],
             self.x0[2],
             self.x0[3] * math.cos(ry),
             self.x0[3] * math.sin(ry),
             self.x0[4] * math.cos(ry),
             self.x0[4] * math.sin(ry),
             self.x0[5]])
        # initialize prior uncertainty
        self.pos_std = [x_std[0], x_std[1], x_std[2]]
        self.dim_std = [x_std[3], x_std[4], x_std[5]]  # assume to be the same as dimension?
        self.ry_std = [x_std[-1]]  # rotation, about 10deg.
        self.Q0_feature = np.diag(1 / np.array([
        self.pos_std[0],
        self.pos_std[1],
        self.pos_std[2],
        self.dim_std[0]*abs(math.cos(ry))+self.x0[3]*self.ry_std[0]*abs(math.sin(ry)),
        self.dim_std[0]*abs(math.sin(ry))+self.x0[3]*self.ry_std[0]*abs(math.cos(ry)),
        self.dim_std[1]*abs(math.cos(ry))+self.x0[4]*self.ry_std[0]*abs(math.sin(ry)),
        self.dim_std[1]*abs(math.sin(ry))+self.x0[4]*self.ry_std[0]*abs(math.cos(ry)),
        self.dim_std[2]])**2)
        self.Q0_X = np.diag(1 / np.array([
        self.pos_std[0],
        self.pos_std[1],
        self.pos_std[2],
        self.dim_std[0],
        self.dim_std[1],
        self.dim_std[2],
        self.ry_std[0]]) ** 2)
        self.posterior = None
        self.mean_LIDAR_variance = None

    @staticmethod
    def feature0_to_x0(feature0):
        '''
        Convert feature0(8dim) to x0(7dim).
        Args:
            feature0 (8,): [xc, yc, zc, lcosry, lsinry, wcosry, wsinry, h]
        Return:
            x0 (7,): [xc, yc, zc, l, w, h, ry]
        '''
        feature0 = feature0.reshape(-1)
        if feature0.shape[0] == 7:
            return feature0
        xc, yc, zc, lcosry, lsinry, wcosry, wsinry, h = feature0[:]
        l = (lcosry **2 + lsinry **2) **0.5
        w = (wcosry **2 + wsinry **2) **0.5
        tanry1 = lsinry/lcosry
        ry1 = np.arctan(tanry1)
        tanry2 = wsinry/wcosry
        ry2 = np.arctan(tanry2)
        return np.array([xc, yc, zc, l, w, h, (ry1 + ry2) * 0.5])

    def set_posterior(self, mean, cov):
        '''
        Args:
            mean (np.array (1,8)/(1,7))
            cov (np.array (8,8)/(7,7))
        '''
        self.posterior = (mean, cov)

    def embedding(self, x0, z_surface):
        '''
        Embed z_surface according to x0
        Args:
            x0: label [xc,yc,zc,l,w,h,ry]
            z_surface (np.array (N, 2)): z_surface is 2D
                dim1: z_surface_bev, from 0 to 4
                ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2, l along 1st axis);
                same to uncertain_label_BEV.embedding.
                dim2: y-axis, from -0.5 to 0.5
                (-h/2 -> h/2)
        Returns:
            z_embed (np.array (N, 3)): 3D points in the object frame (no rotation).
        '''
        z_embed = np.zeros((z_surface.shape[0], 3))
        x0_BEV = x0[[0,1,3,4,6]]
        z_embed[:,[0,1]] = uncertain_label_BEV.embedding(self, x0_BEV, z_surface[:,0])
        z_surface[z_surface[:,1]<-0.5,1] += 1
        z_surface[z_surface[:,1]> 0.5,1] -= 1
        z_embed[:,2] = z_surface[:,1]*x0[5]
        return z_embed

    def project(self, x0, z_embed):
        '''
        Project z_embed to z_surface according to x0
        Args:
            x0: label [xc,yc,zc,l,w,h,ry]
            z_embed: (np.array (N, 3)): 3D points in the object frame (no rotation).
        Returns:
            z_surface (np.array (N, 2)): z_surface is 2D
                dim1: z_surface_bev, from 0 to 4
                ([-l,-w]/2->[l,-w]/2->[l,w]/2->[-l,w]/2, l along 1st axis);
                same to uncertain_label_BEV.embedding.
                dim2: y-axis, from -0.5 to 0.5
                (-h/2 -> h/2)
        '''
        z_surface = np.zeros((z_embed.shape[0],2))
        x0_BEV = x0[[0,1,3,4,6]]
        z_surface[:,0] = uncertain_label_BEV.project(self, x0_BEV, z_embed[:,[0,1]])
        z_surface[:,1] = z_embed[:,2]/x0[5]
        return z_surface

    def add_surface_coord(self, z_surface, z_coord):
        z_surface_added = z_surface+z_coord
        z_surface_added[:,1] = np.minimum(-0.5,np.maximum(z_surface_added[:,1],0.5))
        return z_surface_added

    def Jacobian_surface(self, z_surface):
        '''
        Compute Jacobian matrix
        \frac{\partial [x, y, z]}
        {\partial{xc1, yc1, zc1, lcosry, lsinry, wcosry, wsinry, h}}
        Args:
            z_surface (np.ndarray (N, 2))
        Returns:
            J (np.ndarry (N, 3, 8))
        '''
        # parameterize surface point w.r.t. BEV box feature vector
        z_normed = self.embedding(self.x0, z_surface) / self.x0[3:6]
        Jacobian = self.Jacobian_z_normed(z_normed)
        return Jacobian

    def Jacobian_z_normed_x0(self, z_normed):
        '''
        Compute Jacobian matrix
        \frac{\partial [x, y, z]}
        {\partial{xc, yc, zc, l, w, h, ry}}
        Args:
            z_normed (np.ndarray (N,3))
        Returns:
            J (np.ndarray (N, 3, 8))
        Notes:
        The relationship is
        [x] = [lcosry, -wsinry, 0] [z_norm0] + [xc]
        [y]   [lsinry, wcosry , 0] [z_norm1]   [yc]
        [z]   [0     , 0      , h] [z_norm2]   [zc]
        '''
        return jiou_utils.jacobian_z_normed_x0_3d(self.x0, z_normed)

    def Jacobian_z_normed(self, z_normed):
        '''
        Compute Jacobian matrix
        \frac{\partial [x, y, z]}
        {\partial{xc1, yc1, zc1, lcosry, lsinry, wcosry, wsinry, h}}
        Args:
            z_normed (np.ndarray (N,3))
        Returns:
            J (np.ndarray (N, 3, 8))
        Notes:
        The relationship is
        [x] = [lcosry, -wsinry, 0] [z_norm0] + [xc]
        [y]   [lsinry, wcosry , 0] [z_norm1]   [yc]
        [z]   [0     , 0      , h] [z_norm2]   [zc]
        '''
        return jiou_utils.jacobian_z_normed_feat0_3d(z_normed)

    def sample_prob(self, points, sample_grid=0.1):
        '''
        Sample probability given points.
        Args:
            points(np.ndarray (N, 3)): 3D coordinates
            sample_grid (float): resolution of the sample grid.
        Returns:
            probs (np.ndarray (N)): probability of points in the posterior
            distribution.
        '''
        return jiou_utils.sample_prob_3d(points,
            mean=self.x0, cov=self.posterior[1],
            sample_grid=sample_grid)

    def calc_uncertainty_points(self, z_normed):
        '''
        Propagate the uncertainty from label posterior distribution to 3D points.
        Args:
            z_normed (np.ndarray (N, 3))
        Returns:
            mean (np.ndarray (N, 3))
            cov (np.ndarray (N, 3, 3))
        '''
        return jiou_utils.calc_uncertainty_points_3d(z_normed,
            mean=self.x0, cov=self.posterior[1])
