'''
Adapted from
https://github.com/ZiningWang/Inferring-Spatial-Uncertainty-in-Object-Detection
'''
import numpy as np
from .label import uncertain_label_BEV, uncertain_label_3D

class label_inference_BEV:
    '''
    This class is for inferring BEV spatial uncertainty with a generative model.
    '''
    def __init__(self,
        degree_register=0,
        gen_std=0.2,
        prob_outlier=0.1,
        eps_prior_scaler=0.04,
        boundary_sample_interval=0.1):
        '''
        Args:
            degree_register (int): it should be 1
            gen_std (float): GMM std, in meter
            prob_outlier (float):
            eps_prior_scaler (float): It is to prevent deprecated posterior variance.
                It the posterior covariance is not full rank, add some epsilon of prior.
            boundary_sample_interval (float): interval for sampling along the boundary
                (in the z space).
        '''
        self.generator_std = gen_std
        self.boundary_sample_interval = boundary_sample_interval
        self.prob_outlier = 2*np.pi*self.generator_std**2*prob_outlier/(1-prob_outlier)*(2*degree_register+1)#prob_outlier * (2*degree_register+1) / 3
        self.eps_prior_scaler = eps_prior_scaler
        # use Variantional Bayes to allow analytical result with multiple register
        self.degree_register = degree_register
        self.z_coords = [z_coord * self.boundary_sample_interval for z_coord in
                            range(-self.degree_register, self.degree_register + 1)]
        self.num_z_y = 1 + 2 * degree_register  # maximum number of registered z given y
        self.feature_dim = 6
        self.space_dim = 2 
        self.surface_dim = self.space_dim-1

    def generator(self, z_surface, y):
        # modeled as Gaussian, only store the first and second order term (discard constant term)
        ny = y.shape[0]
        K = z_surface.shape[2]
        prob_y_x_quad = np.zeros((ny, K, self.feature_dim, self.feature_dim))
        prob_y_x_lin = np.zeros((ny, K, 1, self.feature_dim))
        var_generator = self.generator_std ** 2
        for ik in range(K):
            Jacobians_point = self.label.Jacobian_surface(z_surface[:, :, ik])
            # norm(y-Jacobian*x_feature, 2)/self.generator_std^2
            for iy in range(ny):
                Jacobian_point = np.squeeze(Jacobians_point[iy, :, :])
                prob_y_x_quad[iy, ik, :, :] = np.matmul(Jacobian_point.transpose(), Jacobian_point) / var_generator
                prob_y_x_lin[iy, ik, :, :] = -2 * np.matmul(y[iy, :], Jacobian_point) / var_generator

        return (prob_y_x_quad, prob_y_x_lin)

    def register(self, y, z=None):
        # y: ny*2
        ny = y.shape[0]
        nz = self.num_z_y
        prob_z_y = np.zeros((ny, nz))
        yz_distances = np.zeros((ny,nz))
        z_surface_out = np.zeros((ny,self.surface_dim, nz))

        yc = np.matmul(y - self.label.x0[0:self.space_dim], self.label.rotmat)
        zc = yc / self.label.x0[self.space_dim:2*self.space_dim]  # transposed twice
        zc_norm = np.abs(zc)
        z_embed = zc / zc_norm.max(1, keepdims=True) * self.label.x0[self.space_dim:2*self.space_dim] / 2  # z in 3D space, map to surface
        z_surface = self.label.project(self.label.x0, z_embed)  # z in surface_manifold

        var_generator = self.generator_std ** 2
        for iz, z_coord in enumerate(self.z_coords):
            z_surface_out[:, :, iz] = self.label.add_surface_coord(z_surface, z_coord).reshape((ny,self.surface_dim))
            z_embeded = self.label.embedding(self.label.x0, z_surface_out[:,:,iz])
            # Nobody: fix it to adopt GMM.
            # yz_distances[:, iz] = np.linalg.norm(yc - z_embed, 2, axis=1)
            yz_distances[:, iz] = np.linalg.norm(yc - z_embeded, 2, axis=1)
            prob_z_y[:, iz] = np.exp(-yz_distances[:, iz]**2 / 2 / var_generator)

        prob_inlier = np.sum(prob_z_y, axis=1)
        for iz in range(len(self.z_coords)):
            prob_z_y[:, iz] = prob_z_y[:, iz] / (prob_inlier + self.prob_outlier)
        return prob_z_y, z_surface_out, yz_distances

    def infer(self, y, boxBEV, prior_scaler=0):
        # only one z possible z for each y, so prob_z_y no need
        self.label = self.construct_uncertain_label(boxBEV) 
        if y.shape[0] < self.space_dim+1:
            #too few observations, directly use prior
            self.label.set_posterior(self.label.x0, self.return_prior_precision_matrix())
            return self.label
        else:
            converge_checker = ConvergeChecker(self.label.x0, eps=1e-1, max_iter=30)
            while not converge_checker.is_converge:
                prob_z_y, z_surface, yz_distances = self.register(y)
                prob_y_x_quad, prob_y_x_lin = self.generator(z_surface, y)
                ny = prob_y_x_quad.shape[0]
                # get Gaussian of feature vector
                Q = np.squeeze(np.tensordot(prob_z_y, prob_y_x_quad, axes=((0, 1), (0, 1))))  # self.feature_dim*self.feature_dim
                P = np.tensordot(prob_z_y, prob_y_x_lin, axes=((0, 1), (0, 1))).reshape((self.feature_dim, 1))
                Q += prior_scaler*self.label.Q0_feature #scale the prior of variance, default is 0, which is no prior
                #Nobody: add prior to the linear term.
                P -= 2*prior_scaler * self.label.Q0_feature @ self.label.feature0.reshape(-1, 1)
                if np.linalg.matrix_rank(Q,tol=2.5e-3) < Q.shape[0]:
                    Q += self.eps_prior_scaler * self.label.Q0_feature
                post_mean, post_cov = np.linalg.solve(-2 * Q, P), np.linalg.inv(Q)
                self.label.set_posterior(post_mean, post_cov)
                self.label.x0 = self.label.feature0_to_x0(post_mean)
                converge_checker.update(self.label.x0)
            #estimate the LIDAR measurement noise
            mean_variance = np.sum(np.sum(prob_z_y*yz_distances**2, axis=1), axis=0)/np.sum(np.sum(prob_z_y, axis=1),axis=0)/2
            self.label.mean_LIDAR_variance = mean_variance
            return self.label

    def return_prior_precision_matrix(self):
        return np.linalg.inv(self.eps_prior_scaler*self.label.Q0_X)

    def construct_uncertain_label(self, boxBEV):
        return uncertain_label_BEV(boxBEV)

class label_inference_3D(label_inference_BEV):
    '''
    This class is for inferring 3D spatial uncertainty with a generative model.
    '''
    def __init__(self,
        degree_register=0,
        gen_std=0.2,
        prob_outlier=0.1,
        eps_prior_scaler=0.04,
        boundary_sample_interval=0.1):
        '''
        Args:
            degree_register (int): it should be 1
            gen_std (float): GMM std, in meter
            prob_outlier (float):
            eps_prior_scaler (float): It is to prevent deprecated posterior variance.
                It the posterior covariance is not full rank, add some epsilon of prior.
            boundary_sample_interval (float): interval for sampling along the boundary
                (in the z space).
        '''
        label_inference_BEV.__init__(self,
            degree_register,
            gen_std,
            prob_outlier,
            eps_prior_scaler,
            boundary_sample_interval)
        self.z_coords = []
        for zBEV_coord in range(-self.degree_register, self.degree_register + 1):
            self.z_coords.append(np.array([
                zBEV_coord*self.boundary_sample_interval, 0]))
        for zH_coord in range(-self.degree_register, 0):
            self.z_coords.append(np.array([
                0, zH_coord*self.boundary_sample_interval]))
        for zH_coord in range(1, self.degree_register + 1):
            self.z_coords.append(np.array([
                0, zH_coord*self.boundary_sample_interval]))

        self.num_z_y = 1 + 4 * degree_register  # neighborhood points in 2D surface (axis aligned)
        self.feature_dim = 8 #BEV + y center + height
        self.space_dim = 3
        self.surface_dim = self.space_dim-1

    def infer(self, y, box3D, prior_scaler=0):
        return super().infer(y, box3D, prior_scaler)

    def return_prior_precision_matrix(self):
        return np.linalg.inv(self.eps_prior_scaler*self.label.Q0_feature)

    def construct_uncertain_label(self, box3D):
        return uncertain_label_3D(box3D)

class ConvergeChecker:
    '''
    This class is used to check convergence.
    '''
    def __init__(self, x0, eps, max_iter):
        '''
        Args:
            x0 (np.ndarray)
            eps (float)
            max_iter (int)
        '''
        self.x0 = x0.reshape(-1)
        self.x0_old = None
        self.eps = eps
        self.iter = 0
        self.max_iter = max_iter

    @property
    def is_converge(self):
        if self.iter == 0:
            return False
        if self.iter >= self.max_iter:
            print(f"Warning: early stop in the {self.max_iter} step")
            return True
        if np.abs(self.x0 - self.x0_old).sum() < self.eps:
            return True
        return False

    def update(self, x0):
        err_msg = f"The argument x0 is expected the same to self.x0"
        assert x0.size == self.x0.size, err_msg
        self.x0_old = self.x0
        self.x0 = x0.reshape(-1)
        self.iter += 1
