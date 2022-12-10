import torch
from torch import nn
from ...utils import bayesian_utils
from .pointpillar import pointpillar_post_processing, \
    outer_bmm, NoScalingDropout
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms

def point_rcnn_bayesian_inference(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior,
            mode="full-net",
            dropout_rate=None):
    if mode == "full-net":
        return point_rcnn_bayesian_inference_fullnet(
                    model,
                    batch_dict,
                    wdist_dict,
                    num_MC_samples,
                    pdist_prior)
    elif mode == "last-module":
        return point_rcnn_bayesian_inference_lastmodule(
                    model,
                    batch_dict,
                    wdist_dict,
                    num_MC_samples,
                    pdist_prior)
    elif mode == "last-layer":
        return point_rcnn_bayesian_inference_lastlayer(
                    model,
                    batch_dict,
                    wdist_dict,
                    num_MC_samples,
                    pdist_prior)
    elif mode == "mcdropout":
        err_msg = "dropout_rate should not be None."
        assert dropout_rate is not None, err_msg
        return point_rcnn_bayesian_inference_mcdropout(
                    model,
                    batch_dict,
                    num_MC_samples,
                    pdist_prior,
                    dropout_rate)
    else:
        err_msg = f"Unrecognized mode: {mode}."
        raise NotImplementedError(err_msg)

def point_rcnn_bayesian_inference_mcdropout(
            model,
            batch_dict,
            num_MC_samples,
            pdist_prior,
            dropout_rate):
    def _add_mcdropout(model):
        roi_head = model.module_list[-1]
        # check whether it already has an MCDropout layer
        has_mcdropout = False
        for _, model in roi_head.named_modules():
            layer_type = model.__class__.__name__
            if "NoScalingDropout" in layer_type and model.p > 0:
                has_mcdropout = True
                break
        if has_mcdropout:
            return
        # if not has_mcdropout, add an MCDropout layer
        l = list(roi_head.cls_layers)
        l.insert(-1, NoScalingDropout(p=dropout_rate))
        roi_head.cls_layers = nn.Sequential(*l)
        l = list(roi_head.reg_layers)
        l.insert(-1, NoScalingDropout(p=dropout_rate))
        roi_head.reg_layers = nn.Sequential(*l)
        return

    modules = model.module_list
    module2name = {module: name for name, module in model.named_children()}
    module_names = [module2name[itm] for itm in modules]
    _add_mcdropout(model)

    # PointNet2MSG & PointHeadBox
    for cur_module in modules[:-1]:
        batch_dict = cur_module(batch_dict)
    # PointRCNNHead
    batch_dict = rcnn_bayesian_inference_mcdropout(
        [modules[-1]], [module_names[-1]],
        batch_dict, num_MC_samples, pdist_prior,
        return_selected=False)
    pred_dicts, recall_dicts = point_rcnn_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def point_rcnn_bayesian_inference_lastlayer(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    modules = model.module_list
    module2name = {module: name for name, module in model.named_children()}
    module_names = [module2name[itm] for itm in modules]

    # PointNet2MSG & PointHeadBox
    for cur_module in modules[:-1]:
        batch_dict = cur_module(batch_dict)
    # PointRCNNHead
    # mask-out all the elements by 0s in wdist_dict["cov_dict"] except:
    keep_cov_keys = ["roi_head.reg_layers.7", "roi_head.cls_layers.7"]
    for k, v in wdist_dict["cov_dict"].items():
        if all([itm not in k for itm in keep_cov_keys]):
            wdist_dict["cov_dict"][k] = torch.zeros_like(v)
    batch_dict = rcnn_bayesian_inference(
        [modules[-1]], [module_names[-1]],
        batch_dict, wdist_dict,
        num_MC_samples, pdist_prior, return_selected=False)
    pred_dicts, recall_dicts = point_rcnn_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def point_rcnn_bayesian_inference_lastmodule(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    '''
    Bayesian inference for PointRCNN.
    It adopts Monte-Carlo estimator (last module) to approximate
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
    modules = model.module_list
    module2name = {module: name for name, module in model.named_children()}
    module_names = [module2name[itm] for itm in modules]

    # PointNet2MSG & PointHeadBox
    for cur_module in modules[:-1]:
        batch_dict = cur_module(batch_dict)
    # PointRCNNHead
    batch_dict = rcnn_bayesian_inference(
        [modules[-1]], [module_names[-1]],
        batch_dict, wdist_dict,
        num_MC_samples, pdist_prior, return_selected=False)
    pred_dicts, recall_dicts = point_rcnn_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def point_rcnn_bayesian_inference_fullnet(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    '''
    Bayesian inference for PointRCNN.
    It adopts Monte-Carlo estimator (full net) to approximate
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
    modules = model.module_list
    module2name = {module: name for name, module in model.named_children()}
    module_names = [module2name[itm] for itm in modules]

    # PointNet2MSG & PointHeadBox
    batch_dict, batch_dict_list_rpn = rpn_bayesian_inference(
        modules[:2], module_names[:2],
        batch_dict, wdist_dict,
        num_MC_samples, return_list=True)
    # PointRCNNHead
    batch_dict, selected_rcnn = rcnn_bayesian_inference(
        [modules[-1]], [module_names[-1]],
        batch_dict, wdist_dict,
        num_MC_samples, pdist_prior, return_selected=True)
    batch_dict = merge_covs(batch_dict_list_rpn, selected_rcnn, batch_dict)
    pred_dicts, recall_dicts = point_rcnn_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def merge_covs(batch_dict_list_rpn, selected_rcnn, batch_dict):
    '''
    Merge the batch_box_cov in rpn and rcnn.
    Args:
        batch_dict_list_rpn (List[Dict]): MC samples of rpn.
            - "batch_box_preds"
        selected_rcnn : <selected> of rcnn (refer to ROIHeadTemplate.proposal_layer)
        batch_dict: results of bayesian inference after rcnn
            - "batch_box_cov"
    Returns:
        batch_dict: update the "batch_box_cov"
    '''
    rpn_list = []
    for batch_dict_rpn in batch_dict_list_rpn:
        batch_size = int(batch_dict_rpn["batch_index"].max().item())+1
        batch_box_preds_list = []
        for i in range(batch_size):
            batch_mask = batch_dict_rpn["batch_index"] == i
            batch_box_preds = batch_dict_rpn["batch_box_preds"][batch_mask]
            batch_box_preds = batch_box_preds[selected_rcnn[i]]
            batch_box_preds_list.append(batch_box_preds.unsqueeze(0))
        rpn_list.append({
            "batch_box_preds": torch.cat(batch_box_preds_list, dim=0),
        })
    _, cov = compute_mean_and_cov(rpn_list, "batch_box_preds", 0)
    batch_dict["batch_box_cov"] += cov
    return batch_dict

def rpn_bayesian_inference(
    modules,
    module_names,
    batch_dict,
    wdist_dict,
    num_MC_samples,
    return_list=False):
    '''
    Bayesian inference RPN with a Monte-Carlo estimator.
    Update batch_dict with MC estimations:
    - batch_cls_preds (mean)
    - batch_box_preds (mean)
    - point_features (mean)
    - point_cls_scores (mean)
    Add the following results to batch_dict:
    - point_coords
    - batch_index
    - cls_preds_normalized
    Return the batch_dict_list additionally
    '''
    # MC estimate
    batch_dict_list = []
    collect_keys = ["point_features", "point_coords",
        "point_cls_scores", "batch_cls_preds", "batch_box_preds",
        "batch_index", "cls_preds_normalized"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    for _ in range(num_MC_samples):
        for i, (module, name) in enumerate(zip(modules, module_names)):
            bayesian_utils.sample_from_weight_distribution(
                module,
                wdist_dict['mean_dict'],
                wdist_dict['cov_dict'],
                modify_key_fn=lambda k: f"{name}."+k)
            batch_dict_ = module(batch_dict) if i == 0 \
                     else module(batch_dict_)
        collect_fn(batch_dict_)
    # compute mean and update batch_dict
    batch_dict.update({
        "point_features": compute_mean(batch_dict_list, "point_features"),
        "point_cls_scores": compute_mean(batch_dict_list, "point_cls_scores"),
        "batch_cls_preds": compute_mean(batch_dict_list, "batch_cls_preds"),
        "batch_box_preds": compute_mean(batch_dict_list, "batch_box_preds"),
        "point_coords": batch_dict_list[0]["point_coords"],
        "batch_index": batch_dict_list[0]["batch_index"],
        "cls_preds_normalized": batch_dict_list[0]["cls_preds_normalized"]
    })
    return (batch_dict, batch_dict_list) if return_list else batch_dict

def rcnn_bayesian_inference(
    modules,
    module_names,
    batch_dict,
    wdist_dict,
    num_MC_samples,
    pdist_prior,
    return_selected=False):
    '''
    Bayesian inference RPN with a Monte-Carlo estimator.
    Update batch_dict with MC estimations:
    - batch_cls_preds (mean)
    - batch_box_preds (mean)
    Add the following results to batch_dict:
    - rois
    - roi_scores
    - roi_labels
    - has_class_labels
    Compute the covariance matrices and add them to batch_dict:
    - batch_cls_preds
    - batch_box_preds
    Return the selected additionally
    '''
    selected = _get_selected_rcnn(batch_dict, modules[0].model_cfg.NMS_CONFIG['TEST'])
    # MC estimate
    batch_dict_list = []
    collect_keys = ["batch_cls_preds", "batch_box_preds",
        "rois", "roi_scores", "roi_labels", "has_class_labels"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    for _ in range(num_MC_samples):
        for i, (module, name) in enumerate(zip(modules, module_names)):
            bayesian_utils.sample_from_weight_distribution(
                module,
                wdist_dict['mean_dict'],
                wdist_dict['cov_dict'],
                modify_key_fn=lambda k: f"{name}."+k)
            batch_dict_ = module(batch_dict) if i == 0 \
                     else module(batch_dict_)
        collect_fn(batch_dict_)
    # compute mean and cov
    mean, cov = compute_mean_and_cov(
        batch_dict_list, "batch_cls_preds", pdist_prior["batch_cls_preds"])
    batch_dict.update({
        "batch_cls_preds": mean,
        "batch_cls_mean": mean,
        "batch_cls_cov": cov})
    mean, cov = compute_mean_and_cov(
        batch_dict_list, "batch_box_preds", pdist_prior["batch_box_preds"])
    batch_dict.update({
        "batch_box_preds": mean,
        "batch_box_mean": mean,
        "batch_box_cov": cov})
    # update batch_dict
    batch_dict.update({
        "rois": batch_dict_list[0]["rois"],
        "roi_scores": batch_dict_list[0]["roi_scores"],
        "roi_labels": batch_dict_list[0]["roi_labels"],
        "has_class_labels": batch_dict_list[0]["has_class_labels"],
    })
    return (batch_dict, selected) if return_selected else batch_dict

def rcnn_bayesian_inference_mcdropout(
    modules,
    module_names,
    batch_dict,
    num_MC_samples,
    pdist_prior,
    return_selected=False):
    '''
    Bayesian inference RPN with a Monte-Carlo estimator.
    It assumes <modules> already contains at least one Dropout layer which dropout rate < 1.
    Update batch_dict with MC estimations:
    - batch_cls_preds (mean)
    - batch_box_preds (mean)
    Add the following results to batch_dict:
    - rois
    - roi_scores
    - roi_labels
    - has_class_labels
    Compute the covariance matrices and add them to batch_dict:
    - batch_cls_preds
    - batch_box_preds
    Return the selected additionally
    '''
    selected = _get_selected_rcnn(batch_dict, modules[0].model_cfg.NMS_CONFIG['TEST'])
    # MC estimate
    batch_dict_list = []
    collect_keys = ["batch_cls_preds", "batch_box_preds",
        "rois", "roi_scores", "roi_labels", "has_class_labels"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    for _ in range(num_MC_samples):
        for i, (module, _) in enumerate(zip(modules, module_names)):
            batch_dict_ = module(batch_dict) if i == 0 \
                     else module(batch_dict_)
        collect_fn(batch_dict_)
    # compute mean and cov
    mean, cov = compute_mean_and_cov(
        batch_dict_list, "batch_cls_preds", pdist_prior["batch_cls_preds"])
    batch_dict.update({
        "batch_cls_preds": mean,
        "batch_cls_mean": mean,
        "batch_cls_cov": cov})
    mean, cov = compute_mean_and_cov(
        batch_dict_list, "batch_box_preds", pdist_prior["batch_box_preds"])
    batch_dict.update({
        "batch_box_preds": mean,
        "batch_box_mean": mean,
        "batch_box_cov": cov})
    # update batch_dict
    batch_dict.update({
        "rois": batch_dict_list[0]["rois"],
        "roi_scores": batch_dict_list[0]["roi_scores"],
        "roi_labels": batch_dict_list[0]["roi_labels"],
        "has_class_labels": batch_dict_list[0]["has_class_labels"],
    })
    return (batch_dict, selected) if return_selected else batch_dict

def point_rcnn_post_processing(model, batch_dict):
    '''
    Same to pointpillar.
    The "roi" has also been handeled well in pointpillar_post_processing.
    '''
    return pointpillar_post_processing(model, batch_dict)

def compute_mean(batch_dict_list, key):
    v = torch.cat([
        batch_dict[key].unsqueeze(0)
        for batch_dict in batch_dict_list], dim=0)
    mean = v.mean(dim=0)
    return mean

def compute_mean_and_cov(batch_dict_list, key, pdist_prior):
    mean = compute_mean(batch_dict_list, key)
    # cov = 2nd-moment(mean2) + prior - mean * mean
    dim = mean.shape[-1]
    mean2 = []
    for batch_dict in batch_dict_list:
        mean2_ = outer_bmm(batch_dict[key], batch_dict[key])
        mean2.append(mean2_.unsqueeze(0))
    mean2 = torch.cat(mean2, dim=0).mean(dim=0)
    # batched identity matrix
    batched_I = torch.eye(dim, device=mean2.device)\
        .reshape(1, 1, dim, dim)\
        .repeat(*mean2.shape[:-2], 1, 1)
    pdist_prior *= batched_I
    cov = mean2 + pdist_prior - outer_bmm(mean, mean)
    return mean, cov

def _get_selected_rcnn(batch_dict, nms_config):
    """
    Args:
        batch_dict:
            batch_size:
            batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
            batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
            cls_preds_normalized: indicate whether batch_cls_preds is normalized
            batch_index: optional (N1+N2+...)
        nms_config:

    Returns:
        batch_dict:
            rois: (B, num_rois, 7+C)
            roi_scores: (B, num_rois)
            roi_labels: (B, num_rois)

    """
    if batch_dict.get('rois', None) is not None:
        return batch_dict
        
    batch_size = batch_dict['batch_size']
    batch_box_preds = batch_dict['batch_box_preds']
    batch_cls_preds = batch_dict['batch_cls_preds']
    batch_selected = []

    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_cls_preds.shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_cls_preds'].shape.__len__() == 3
            batch_mask = index
        box_preds = batch_box_preds[batch_mask]
        cls_preds = batch_cls_preds[batch_mask]

        cur_roi_scores, _ = torch.max(cls_preds, dim=1)

        if nms_config.MULTI_CLASSES_NMS:
            raise NotImplementedError
        else:
            selected, _ = class_agnostic_nms(
                box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
            )
            batch_selected.append(selected.unsqueeze(0))
    return torch.cat(batch_selected, dim=0)
