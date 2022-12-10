import numpy as np
from pathlib import Path
from det3.ops import write_pkl

def is_cov_vec(covs):
    '''
    Return True, if <covs> is in vector format.
    Args
        covs: (Tensor)
    '''
    return len(covs.shape) == 2

def is_cov_mat(covs):
    '''
    Return True, if <covs> is in matrix format.
    Args
        covs: (Tensor)
    '''
    return len(covs.shape) == 3

def generate_predictive_distribution_kitti(
    batch_dict,
    pred_dicts,
    class_names,
    output_path=None):
    '''
    Generate predictive distributions for the KITTI dataset.
    Args:
        batch_dict:
            frame_id:
        pred_dicts: list of pred_dicts
            pred_boxes: (N, 7), Tensor
            pred_scores: (N), Tensor
            pred_labels: (N), Tensor
            pdist_boxes_mean: (N, 7), pdist_boxes_mean, 
            pdist_boxes_cov: (N, 7)/ (N, 7, 7), pdist_boxes_cov,
        class_names:
        output_path:
    Returns:
        pdist (List[Dict])
    TODO: The pred_boxes are expected the same as the pred_boxes_mean
        (check by assertion).
    '''
    for index, pred_dict in enumerate(pred_dicts):
        frame_id = batch_dict['frame_id'][index]
        d3_means = pred_dict['pdist_boxes_mean'].cpu().numpy().reshape(-1, 7)
        d3_covs = pred_dict['pdist_boxes_cov'].cpu().numpy()
        N = d3_means.shape[0]
        if is_cov_vec(d3_covs):
            d3_covs = d3_covs.reshape(-1, 7)
            assert d3_covs.shape[0] == N
            pdist = []
            for d3_mean, d3_cov in zip(d3_means, d3_covs):
                bev_mean = d3_mean.reshape(-1)[[0,1,3,4,6]]
                bev_cov  =  d3_cov.reshape(-1)[[0,1,3,4,6]]
                pdist_ = {
                    "post_mean_bev": bev_mean.reshape(-1, 5),
                    "post_cov_bev": np.diag(bev_cov).reshape(5,5),
                    "post_mean_3d": d3_mean.reshape(-1, 7),
                    "post_cov_3d": np.diag(d3_cov).reshape(7,7),
                }
                pdist.append(pdist_)
        elif is_cov_mat(d3_covs):
            d3_covs = d3_covs.reshape(-1, 7, 7)
            assert d3_covs.shape[0] == N
            pdist = []
            for d3_mean, d3_cov in zip(d3_means, d3_covs):
                bev_mean = d3_mean.reshape(-1)[[0,1,3,4,6]]
                bev_cov = np.concatenate(
                    [d3_cov[i, [0,1,3,4,6]].reshape(1, 5)
                    for i in [0,1,3,4,6]], axis=0)
                pdist_ = {
                    "post_mean_bev": bev_mean.reshape(-1, 5),
                    "post_cov_bev": bev_cov.reshape(5,5),
                    "post_mean_3d": d3_mean.reshape(-1, 7),
                    "post_cov_3d": d3_cov.reshape(7,7),
                }
                pdist.append(pdist_)
        else:
            raise NotImplementedError
        if output_path is not None:
            write_pkl(pdist, Path(output_path)/f"{frame_id}.pkl")
    return pdist
