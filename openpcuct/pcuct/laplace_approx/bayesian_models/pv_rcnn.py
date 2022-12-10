from .point_rcnn import point_rcnn_bayesian_inference

def pv_rcnn_bayesian_inference(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    '''
    Bayesian inference for PVRCNN.
    It adopts Monte-Carlo estimator to approximate
    the predictive distribution. and return it in
    <pred_dicts>.
    Args:
        model: nn.Module
        batch_dict: Dict
        wdist_dict: Dict
        num_MC_samples: int
        pdist_prior: Dict: prior predictive distribution.
    The covariance matrix of the predictive distribution is 
    # cov = 2nd-moment(mean2) + prior - mean * mean
    e.g.{"batch_cls_preds": 1e-4, "batch_box_preds": 1e-4}
    Returns:
        pred_dicts, recall_dicts
    '''
    # Its implementation is the same as point_rcnn.
    return point_rcnn_bayesian_inference(model, batch_dict, wdist_dict, num_MC_samples, pdist_prior)
