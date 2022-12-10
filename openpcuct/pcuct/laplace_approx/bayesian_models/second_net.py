from .pointpillar import pointpillar_bayesian_inference

def second_net_bayesian_inference(
    model,
    batch_dict,
    wdist_dict,
    num_MC_samples,
    pdist_prior,
    mode="full-net",
    dropout_rate=None):
    '''
    Bayesian inference for SECONDNet.
    Args:
        model: nn.Module
        batch_dict: Dict
        wdist_dict: Dict
        num_MC_samples: int
        pdist_prior: Dict: prior predictive distribution.
            The covariance matrix of the predictive distribution is 
            # cov = 2nd-moment(mean2) + prior - mean * mean
            e.g.{"batch_cls_preds": 1e-4, "batch_box_preds": 1e-4}
        mode: str: "full-net"/"last-layer"/"last-module"/"mcdropout"
        dropout_rate: float: dropout rate for MCDropout baseline (set as 1.0 will zero-out all elements).
    Returns:
        pred_dicts, recall_dicts
    '''
    # Its implementation is the same as pointpillar.
    return pointpillar_bayesian_inference(
        model, batch_dict, wdist_dict,
        num_MC_samples, pdist_prior, mode,
        dropout_rate)
