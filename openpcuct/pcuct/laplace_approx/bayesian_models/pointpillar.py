import torch
from torch import nn
from ...utils import bayesian_utils
from pcdet.models.model_utils import model_nms_utils

def pointpillar_bayesian_inference(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior,
            mode="full-net",
            dropout_rate=None):
    '''
    Bayesian inference for PointPillar.
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
        mode: str: "full-net"/"last-layer"/"last-module"/"mcdropout"
        dropout_rate: float: dropout rate for MCDropout baseline (set as 1.0 will zero-out all elements).
    Returns:
        pred_dicts, recall_dicts
    '''
    if mode == "full-net":
        return pointpillar_bayesian_inference_fullnet(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior)
    elif mode in ["last-layer", "last-module"]:
        return pointpillar_bayesian_inference_lastlayer(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior)
    elif mode == "mcdropout":
        err_msg = "dropout_rate should not be None."
        assert dropout_rate is not None, err_msg
        return pointpillar_bayesian_inference_mcdropout(
            model,
            batch_dict,
            num_MC_samples,
            pdist_prior,
            dropout_rate)
    else:
        err_msg = f"Unrecognized mode: {mode}."
        raise NotImplementedError(err_msg)

def pointpillar_bayesian_inference_mcdropout(
            model,
            batch_dict,
            num_MC_samples,
            pdist_prior,
            dropout_rate):
    '''
    Bayesian inference for PointPillar.
    It adopts Monte-Carlo estimator and Dropout to approximate
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
        dropout_rate: float: dropout rate for MCDropout baseline (set as 1.0 will zero-out all elements).
    Returns:
        pred_dicts, recall_dicts
    '''
    def _add_mcdropout(model):
        dense_head = model.module_list[-1]
        # check whether it already has an MCDropout layer
        has_mcdropout = False
        for _, model in dense_head.named_modules():
            layer_type = model.__class__.__name__
            if "NoScalingDropout" in layer_type:
                has_mcdropout = True
                break
        if has_mcdropout:
            return
        # if not has_mcdropout, add an MCDropout layer
        dense_head.conv_cls = nn.Sequential(*[
            NoScalingDropout(p=dropout_rate),
            dense_head.conv_cls])
        dense_head.conv_box = nn.Sequential(*[
            NoScalingDropout(p=dropout_rate),
            dense_head.conv_box])
        if dense_head.conv_dir_cls is not None:
            dense_head.conv_dir_cls = nn.Sequential(*[
                NoScalingDropout(p=dropout_rate),
                dense_head.conv_dir_cls])
        return

    modules = model.module_list
    _add_mcdropout(model)
    batch_dict_list = []
    collect_keys = ["batch_cls_preds", "batch_box_preds"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    # forward network (before Head)
    # 'vfe', 'map_to_bev_module', 'backbone_2d',
    for cur_module in modules[:-1]:
        batch_dict = cur_module(batch_dict)
    # for _ in range(num_MC_samples):
    ## bayesian inference
    dense_head = model.module_list[-1]
    for _ in range(num_MC_samples):
        batch_dict_ = dense_head(batch_dict)
        collect_fn(batch_dict_)
    pdist_mean, pdist_cov = _estimate_predictive_distribution(
        batch_dict_list, pdist_prior)
    batch_dict.update({
            "batch_cls_preds": pdist_mean['batch_cls_preds'],
            "batch_box_preds": pdist_mean['batch_box_preds'],
            "batch_cls_mean": pdist_mean['batch_cls_preds'],
            "batch_cls_cov": pdist_cov['batch_cls_preds'],
            "batch_box_mean": pdist_mean['batch_box_preds'],
            "batch_box_cov": pdist_cov['batch_box_preds'],
    })

    pred_dicts, recall_dicts = pointpillar_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def pointpillar_bayesian_inference_lastlayer(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    '''
    Bayesian inference for PointPillar.
    It adopts Monte-Carlo estimator (last-layer only) to approximate
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

    batch_dict_list = []
    collect_keys = ["batch_cls_preds", "batch_box_preds"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    # forward network (before Head)
    # 'vfe', 'map_to_bev_module', 'backbone_2d',
    for cur_module in modules[:-1]:
        batch_dict = cur_module(batch_dict)
    # for _ in range(num_MC_samples):
    ## sample last layer
    ## bayesian inference
    dense_head = model.module_list[-1]
    dense_head_name = module2name[dense_head]
    for _ in range(num_MC_samples):
        bayesian_utils.sample_from_weight_distribution(
            dense_head,
            wdist_dict["mean_dict"],
            wdist_dict["cov_dict"],
            modify_key_fn=lambda k: f"{dense_head_name}."+k)
        batch_dict_ = dense_head(batch_dict)
        collect_fn(batch_dict_)
    pdist_mean, pdist_cov = _estimate_predictive_distribution(
        batch_dict_list, pdist_prior)
    batch_dict.update({
            "batch_cls_preds": pdist_mean['batch_cls_preds'],
            "batch_box_preds": pdist_mean['batch_box_preds'],
            "batch_cls_mean": pdist_mean['batch_cls_preds'],
            "batch_cls_cov": pdist_cov['batch_cls_preds'],
            "batch_box_mean": pdist_mean['batch_box_preds'],
            "batch_box_cov": pdist_cov['batch_box_preds'],
    })

    pred_dicts, recall_dicts = pointpillar_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def pointpillar_bayesian_inference_fullnet(
            model,
            batch_dict,
            wdist_dict,
            num_MC_samples,
            pdist_prior):
    '''
    Bayesian inference for PointPillar.
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
    batch_dict_list = []
    collect_keys = ["batch_cls_preds", "batch_box_preds"]
    collect_fn = lambda itm: batch_dict_list.append({
        k: itm[k] for k in collect_keys})
    for _ in range(num_MC_samples):
        bayesian_utils.sample_from_weight_distribution(
            model,
            wdist_dict["mean_dict"],
            wdist_dict["cov_dict"])
        for cur_module in model.module_list:
            batch_dict = cur_module(batch_dict)
        collect_fn(batch_dict)
    pdist_mean, pdist_cov = _estimate_predictive_distribution(
        batch_dict_list, pdist_prior)
    batch_dict.update({
            "batch_cls_preds": pdist_mean['batch_cls_preds'],
            "batch_box_preds": pdist_mean['batch_box_preds'],
            "batch_cls_mean": pdist_mean['batch_cls_preds'],
            "batch_cls_cov": pdist_cov['batch_cls_preds'],
            "batch_box_mean": pdist_mean['batch_box_preds'],
            "batch_box_cov": pdist_cov['batch_box_preds'],
    })

    pred_dicts, recall_dicts = pointpillar_post_processing(model, batch_dict)
    return pred_dicts, recall_dicts

def pointpillar_post_processing(model, batch_dict):
    """
    Args:
        model: nn.Module
        batch_dict:
            batch_size:
            batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                            or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
            multihead_label_mapping: [(num_class1), (num_class2), ...]
            batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
            cls_preds_normalized: indicate whether batch_cls_preds is normalized
            batch_index: optional (N1+N2+...)
            has_class_labels: True/False
            roi_labels: (B, num_rois)  1 .. num_classes
            batch_pred_labels: (B, num_boxes, 1)
    Returns:
        pred_dicts, recall_dicts
    """
    post_process_cfg = model.model_cfg.POST_PROCESSING
    batch_size = batch_dict['batch_size']
    recall_dict = {}
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        # box_preds = batch_dict['batch_box_preds'][batch_mask]
        box_preds = batch_dict['batch_box_mean'][batch_mask]
        box_pdists_mean = batch_dict['batch_box_mean'][batch_mask]
        box_pdists_cov = batch_dict['batch_box_cov'][batch_mask]
        src_box_preds = box_preds

        if not isinstance(batch_dict['batch_cls_preds'], list):
            # cls_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_mean'][batch_mask]

            src_cls_preds = cls_preds
            assert cls_preds.shape[1] in [1, model.num_class]
            assert not batch_dict['cls_preds_normalized']
            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
        else:
            # cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
            cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_mean']]
            src_cls_preds = cls_preds
            assert not batch_dict['cls_preds_normalized']
            if not batch_dict['cls_preds_normalized']:
                cls_preds = [torch.sigmoid(x) for x in cls_preds]

        if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
            err_msg = "Do not support MULTI_CLASSES_NMS."
            assert False, err_msg
            if not isinstance(cls_preds, list):
                cls_preds = [cls_preds]
                multihead_label_mapping = [torch.arange(1, model.num_class, device=cls_preds[0].device)]
            else:
                multihead_label_mapping = batch_dict['multihead_label_mapping']

            cur_start_idx = 0
            pred_scores, pred_labels, pred_boxes = [], [], []
            for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                    cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                cur_pred_labels = cur_label_mapping[cur_pred_labels]
                pred_scores.append(cur_pred_scores)
                pred_labels.append(cur_pred_labels)
                pred_boxes.append(cur_pred_boxes)
                cur_start_idx += cur_cls_preds.shape[0]

            final_scores = torch.cat(pred_scores, dim=0)
            final_labels = torch.cat(pred_labels, dim=0)
            final_boxes = torch.cat(pred_boxes, dim=0)
        else:
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds = batch_dict[label_key][index]
            else:
                label_preds = label_preds + 1
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            if post_process_cfg.OUTPUT_RAW_SCORE:
                max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
            pdist_boxes_mean = box_pdists_mean[selected]
            pdist_boxes_cov = box_pdists_cov[selected]

        recall_dict = model.generate_recall_record(
            box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
            recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
            thresh_list=post_process_cfg.RECALL_THRESH_LIST
        )

        make_PSD(pdist_boxes_cov)
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels,
            'pdist_boxes_mean': pdist_boxes_mean, 
            'pdist_boxes_cov': pdist_boxes_cov,
        }
        pred_dicts.append(record_dict)

    return pred_dicts, recall_dict

def _compute_mean_and_cov(batch_dict_list, key, pdist_prior):
    v = torch.cat([
        batch_dict[key].unsqueeze(0)
        for batch_dict in batch_dict_list], dim=0)
    mean = v.mean(dim=0)

    # cov = 2nd-moment(mean2) + prior - mean * mean
    pdist_prior_dict = {
        "batch_cls_preds": pdist_prior.get("batch_cls_preds", 1e-4),
        "batch_box_preds": pdist_prior.get("batch_box_preds", 1e-4),
    }
    dim = mean.shape[-1]
    mean2 = []
    for batch_dict in batch_dict_list:
        mean2_ = outer_bmm(batch_dict[key], batch_dict[key])
        mean2.append(mean2_.unsqueeze(0))
    mean2 = torch.cat(mean2, dim=0).mean(dim=0)
    # batched identity matrix
    batched_I = torch.eye(dim, device=mean2.device)\
        .reshape(1, 1, dim, dim).repeat(*mean2.shape[:-2], 1, 1)
    pdist_prior = pdist_prior_dict[key] * batched_I
    cov = mean2 + pdist_prior - outer_bmm(mean, mean)
    return mean, cov

def outer_bmm(m1, m2):
    '''
    Compute outer product of batched matrix <m1> and <m2>.
    Args:
        m1(torch.Tensor [..., M1])
        m2(torch.Tensor [..., M2])
    Return:
        m(torch.Tensor [..., M1, M2])
    '''
    m1_shape = m1.shape
    m2_shape = m2.shape
    m1_ = m1.reshape(-1, m1_shape[-1], 1)
    m2_ = m2.reshape(-1, 1, m2_shape[-1])
    m = torch.bmm(m1_, m2_)
    return m.reshape(*m1_shape[:-1], m1_shape[-1], m2_shape[-1])

def _estimate_predictive_distribution(batch_dict_list, pdist_prior):
    keys = ["batch_cls_preds", "batch_box_preds"]
    pdist_mean_dict, pdist_cov_dict = {}, {}
    for k in keys:
        mean, cov = _compute_mean_and_cov(batch_dict_list, k, pdist_prior)
        pdist_mean_dict[k] = mean
        pdist_cov_dict[k] = cov
    return pdist_mean_dict, pdist_cov_dict

def make_PSD(m):
    '''
    Return the nearest PSD matrix by replacing the 
    negative eigenvalues with a small positive number.
    Args:
        m (torch.Tensor, [N, D, D])
    Returns:
        (torch.Tensor, [N, D, D])
    '''
    eps = torch.finfo(torch.float32).eps * 10
    L, V = torch.linalg.eig(m)
    L = torch.clamp(L.real, min=eps)
    rtn = V.real @ torch.diag_embed(L) @ torch.linalg.inv(V.real)
    return rtn

class NoScalingDropout(torch.nn.Module):
    '''
    Dropout layer. Different from the conventional Dropout layer,
    this layer works in both training and testing mode and does not
    scaling up/down the elements.
    '''
    def __init__(self, p):
        super().__init__()
        self.p = p
        assert 0 <= p <= 1

    def forward(self, x):
        '''
        Randomly set the elements in x as zeros.
        Args:
            x (torch.Tensor [N,C,*])
        Returns:
            output (torch.Tensor [N,C,*])
        '''
        if self.p == 1:
            return x * torch.zeros_like(x)
        if self.p == 0:
            return x
        noise = torch.bernoulli(torch.ones_like(x) * (1-self.p))
        return x * noise
