import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from det3.ops import write_pkl, read_pkl
from pcdet.models import model_fn_decorator

class DiagonalFisherInformationMatrix:
    def __init__(self):
        self.model = None
        self._data = {}
    
    def register(self, model):
        self.model = model
    
    def update(self, dataloader, closure_fn, setup_fn=lambda model: model):
        '''
        Args:
            setup_fn: lambda_fn model:
                - freeze bn
            closure_fn: lambda_fn model, data: returns loss
                - zero grad
                - compute loss
        Returns:
            cov_dict
        '''
        setup_fn(self.model)
        for k, v in self.model.named_parameters():
            self._data[k] = v.clone().detach().fill_(0)
        for data in tqdm(dataloader):
            det_loss = closure_fn(self.model, data)
            det_loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._data[name] += param.grad.detach().clone() ** 2
        for name, param in self.model.named_parameters():
            self._data[name] /= len(dataloader)
        return self._data

def compute_mean(model):
    '''
    Args:
        model: nn.Module
    Returns:
        mean_dict: Dict
    '''
    # filter out the model parameters that
    # not used in the inference stage
    no_model_keys = [
        "global_step",
    ]
    model_state = deepcopy(model.state_dict())
    [model_state.pop(k) for k in no_model_keys]
    return model_state

def compute_fisher(
    model,
    dataloader,
    closure_fn,
    setup_fn,
    param_dict):
    '''
    Args:
        model: nn.Module
        dataloader: Dataloader
    Returns:
        fisher_dict: Dict
    '''
    fim = DiagonalFisherInformationMatrix()
    fim.register(model)
    fisher_dict = fim.update(dataloader, closure_fn, setup_fn)
    return fisher_dict

def compute_cov_with_diagonal_fisher(
    fisher_dict,
    alpha,
    scalar_prior,
    method):
    '''
    cov = alpha * (fisher + scalar_prior * I)^-1
    Args:
        fisher_dict(Dict[str,torch.Tensor])
        alpha(float)
        scalar_prior(float)
    Return:
        cov_dict(Dict[str,torch.Tensor])
    '''
    if method == "heuristic":
        return compute_cov_with_diagonal_fisher_heuristic(fisher_dict, alpha)
    elif method == "naive":
        return compute_cov_with_diagonal_fisher_naive(fisher_dict, alpha, scalar_prior)
    else:
        raise NotImplementedError

def compute_cov_with_diagonal_fisher_naive(
    fisher_dict,
    alpha,
    scalar_prior):
    '''
    cov = alpha * (fisher + scalar_prior * I)^-1
    Args:
        fisher_dict(Dict[str,torch.Tensor])
        alpha(float)
    Return:
        cov_dict(Dict[str,torch.Tensor])
    '''
    cov_dict = {}
    for k, v in fisher_dict.items():
        cov_dict[k] = alpha / (v + scalar_prior)
    return cov_dict

def compute_cov_with_diagonal_fisher_heuristic(
    fisher_dict,
    alpha):
    '''
    compute threshold with a heuristic method to handling the zero values in fisher..
    alpha = alpha * th
    scaler_prior = th
    cov = alpha * (fisher + scalar_prior * I)^-1
    Args:
        fisher_dict(Dict[str,torch.Tensor])
        alpha(float)
    Return:
        cov_dict(Dict[str,torch.Tensor])
    '''
    def _compute_threshold(v, quantile):
        '''
        Return:
            th (torch.Tensor): the <quantile-th> value, if it is not 0. Otherwise, return the smallest positive value
        '''
        th = v[v>0].flatten().quantile(quantile)
        return th

    cov_dict = {}
    for k, v in fisher_dict.items():
        th = _compute_threshold(v, quantile=0.3)
        # 1*th returns a copy
        scalar_prior = 1 * th
        alpha_ = alpha * (th)
        fix_mask = v < th
        cov_dict[k] = alpha_ / (v + scalar_prior)
        cov_dict[k][fix_mask] = 0
    return cov_dict

def save_weight_distribution(path, mean_dict, fisher_dict):
    '''
    Args:
        path: str
        mean_dict: Dict
        fisher_dict: Dict
    '''
    state = {
        "mean_dict": mean_dict,
        "fisher_dict": fisher_dict
    }
    write_pkl(state, path)

def load_weight_distribution(path, alpha, scalar_prior, method="heuristic"):
    '''
    Args:
        path: str
        method: str "heuristic" or "naive"
    Returns:
        mean_dict: Dict
        cov_dict: Dict
    '''
    state = read_pkl(path)
    fisher_dict = state['fisher_dict']
    cov_dict = compute_cov_with_diagonal_fisher(
        fisher_dict, alpha, scalar_prior, method)
    # cov_dict = compute_cov_with_diagonal_fisher_heuristic(
        # fisher_dict, alpha, mode)
    return state['mean_dict'], cov_dict

def get_setup_fn():
    '''
    Return:
        setup_fn (lambda_fn)
            freeze the batch normalization layers
            - Arg: model
            - Return: None
    '''
    def default_fn(model):
        model.train()
        for _, l in model.named_modules():
            if isinstance(l, nn.BatchNorm1d):
                l.eval()
            elif isinstance(l, nn.BatchNorm2d):
                l.eval()
            elif isinstance(l, nn.BatchNorm3d):
                l.eval()
    setup_fn = default_fn
    return setup_fn

def get_closure_fn(name, mode):
    if mode == "empirical":
        return get_empirical_closure_fn()
    elif mode == "standard":
        if name in ["pointpillar", "second"]:
            return get_standard_pointpillar_closure_fn()
        elif name in ["pointrcnn"]:
            return get_standard_pointrcnn_closure_fn()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def get_empirical_closure_fn():
    '''
    Return:
        closure_fn (lambda_fn):
            clear model gradient, compute <loss> and return it.
            - Args: model, data
            - Return: loss
    '''
    model_fn = model_fn_decorator()
    def default_fn(model, data):
        model.zero_grad()
        det_loss, _, _ = model_fn(model, data)
        return det_loss
    closure_fn = default_fn
    return closure_fn

def get_standard_pointpillar_closure_fn():
    '''
    Return:
        closure_fn (lambda_fn):
            clear model gradient, compute <loss> and return it.
            - Args: model, data
            - Return: loss
    '''
    def _sample_from_pdist(model):
        pdist_prior_dict = {
            "cls_preds": 1e-2,
            "box_preds": 1e-4,
            "dir_cls_preds": 1e-2}
        pdist_samples_dict = {}
        dense_head = model.dense_head
        for k in ["cls_preds", "box_preds", "dir_cls_preds"]:
            v = dense_head.forward_ret_dict.get(k, None)
            if k == "dir_cls_preds" and v == None:
                continue
            v = v.detach()
            v += torch.randn_like(v) * pdist_prior_dict[k]
            pdist_samples_dict[k] = v
        return pdist_samples_dict

    def _compute_loss(model, pdist_samples_dict):
        dense_head = model.dense_head
        cls_loss_func = dense_head.cls_loss_func
        reg_loss_func = dense_head.reg_loss_func
        dir_loss_func = dense_head.dir_loss_func
        cls_preds = dense_head.forward_ret_dict['cls_preds']
        box_preds = dense_head.forward_ret_dict['box_preds']
        dir_cls_preds = dense_head.forward_ret_dict.get('dir_cls_preds', None)
        loss_weights_dict = dense_head.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        batch_size = int(cls_preds.shape[0])

        cls_preds = cls_preds.view(batch_size, -1, dense_head.num_class)
        cls_targets = pdist_samples_dict['cls_preds']\
            .view(batch_size, -1, dense_head.num_class)
        cls_weights = torch.ones(*cls_targets.shape[:2], device=cls_targets.device)
        cls_loss = cls_loss_func(cls_preds, cls_targets, weights=cls_weights)
        cls_loss = cls_loss.sum() / batch_size
        cls_loss = cls_loss * loss_weights_dict['cls_weight']
        loss = cls_loss

        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // dense_head.num_anchors_per_location if not dense_head.use_multihead else
                                   box_preds.shape[-1])
        box_reg_targets = pdist_samples_dict["box_preds"].view(*box_preds.shape)
        box_preds_sin, reg_targets_sin = dense_head.add_sin_difference(
            box_preds, box_reg_targets)
        loc_loss_src = reg_loss_func(box_preds_sin, reg_targets_sin)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * loss_weights_dict['loc_weight']
        loss += loc_loss

        if dir_cls_preds is not None:
            dir_targets = pdist_samples_dict["dir_cls_preds"]
            dir_targets = dir_targets.view(
                batch_size, -1, dense_head.model_cfg.NUM_DIR_BINS)
            dir_logits = dir_cls_preds.view(
                batch_size, -1, dense_head.model_cfg.NUM_DIR_BINS)
            dir_weights = torch.ones(*dir_logits.shape[:2], device=dir_targets.device)
            dir_loss = dir_loss_func(dir_logits, dir_targets, dir_weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['dir_weight']
            loss += dir_loss
        return loss

    model_fn = model_fn_decorator()
    def default_fn(model, data):
        with torch.no_grad():
            model_fn(model, data)
        pdist_samples_dict = _sample_from_pdist(model)
        model_fn(model, data)
        loss = _compute_loss(model, pdist_samples_dict)
        model.zero_grad()
        return loss
    closure_fn = default_fn
    return closure_fn

def get_standard_pointrcnn_closure_fn():
    '''
    Return:
        closure_fn (lambda_fn):
            clear model gradient, compute <loss> and return it.
            - Args: model, data
            - Return: loss
    '''
    def _sample_from_pdist(model):
        pdist_prior_dict = {
            "point_cls_preds": 1e-2,
            "point_box_preds": 1e-4,
            "rcnn_cls": 1e-2,
            "rcnn_reg": 1e-4
        }
        pdist_samples_dict = {}
        point_head = model.point_head
        forward_ret_dict = point_head.forward_ret_dict

        point_cls_preds = forward_ret_dict["point_cls_preds"].clone().detach()
        pdist_prior = pdist_prior_dict["point_cls_preds"]
        point_cls_preds += torch.randn_like(point_cls_preds) * pdist_prior
        point_cls_preds = torch.sigmoid(point_cls_preds)
        pdist_samples_dict.update({"one_hot_targets": point_cls_preds})

        point_box_preds = forward_ret_dict["point_box_preds"].clone().detach()
        pdist_prior = pdist_prior_dict["point_box_preds"]
        point_box_preds += torch.randn_like(point_box_preds) * pdist_prior
        point_box_labels = point_box_preds
        pdist_samples_dict.update({"point_box_labels": point_box_labels})

        roi_head = model.roi_head
        forward_ret_dict = roi_head.forward_ret_dict

        rcnn_cls_labels = forward_ret_dict["rcnn_cls"].clone().detach()
        pdist_prior = pdist_prior_dict["rcnn_cls"]
        rcnn_cls_labels += torch.randn_like(rcnn_cls_labels) * pdist_prior
        rcnn_cls_labels = torch.sigmoid(rcnn_cls_labels)
        pdist_samples_dict.update({"rcnn_cls_labels": rcnn_cls_labels.int()})

        reg_targets = forward_ret_dict["rcnn_reg"].clone().detach()
        pdist_prior = pdist_prior_dict["rcnn_reg"]
        reg_targets += torch.randn_like(reg_targets) * pdist_prior
        pdist_samples_dict.update({"reg_targets": reg_targets})

        return pdist_samples_dict

    def _compute_point_loss_cls(point_head, one_hot_targets):
        point_cls_preds = point_head.forward_ret_dict["point_cls_preds"]
        cls_weights = torch.ones(*one_hot_targets.shape[:-1],
            device=one_hot_targets.device)
        loss = point_head.cls_loss_func(
            point_cls_preds,
            one_hot_targets,
            weights=cls_weights)
        loss = loss.sum()
        loss_weights_dict = point_head.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss *= loss_weights_dict["point_cls_weight"]
        return loss

    def _compute_point_loss_box(point_head, point_box_labels):
        point_box_preds = point_head.forward_ret_dict["point_box_preds"]
        point_box_labels = point_box_labels
        reg_weights = torch.ones(*point_box_labels.shape[:-1],
            device=point_box_labels.device)
        loss = point_head.reg_loss_func(
            point_box_preds[None, ...],
            point_box_labels[None, ...],
            weights=reg_weights[None, ...])
        loss = loss.sum()
        loss_weights_dict = point_head.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss *= loss_weights_dict["point_box_weight"]
        return loss

    def _compute_rcnn_loss_cls(roi_head, rcnn_cls_labels):
        loss_cfgs = roi_head.model_cfg.LOSS_CONFIG
        rcnn_cls = roi_head.forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = rcnn_cls_labels
        rcnn_cls_flat = rcnn_cls.view(-1)
        batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float().view(-1), reduction='none')
        cls_valid_mask = torch.ones_like(rcnn_cls_flat)
        rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        return rcnn_loss_cls
    
    def _compute_rcnn_loss_reg(roi_head, reg_targets):
        loss_cfgs = roi_head.model_cfg.LOSS_CONFIG
        rcnn_reg = roi_head.forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        rcnn_batch_size = rcnn_reg.shape[0]
        reg_valid_mask = torch.ones((rcnn_batch_size), device=rcnn_reg.device)
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        rcnn_loss_reg = roi_head.reg_loss_func(
            rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
            reg_targets.unsqueeze(dim=0),
        )  # [B, M, 7]
        rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
        rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
        return rcnn_loss_reg

    def _compute_loss(model, pdist_samples_dict):
        point_loss_cls = _compute_point_loss_cls(
            model.point_head, pdist_samples_dict["one_hot_targets"])
        loss = point_loss_cls
        point_loss_box = _compute_point_loss_box(
            model.point_head, pdist_samples_dict["point_box_labels"])
        loss += point_loss_box
        rcnn_loss_cls = _compute_rcnn_loss_cls(
            model.roi_head, pdist_samples_dict["rcnn_cls_labels"])
        loss += rcnn_loss_cls
        rcnn_loss_reg = _compute_rcnn_loss_reg(
            model.roi_head, pdist_samples_dict["reg_targets"])
        loss += rcnn_loss_reg
        return loss

    model_fn = model_fn_decorator()
    def default_fn(model, data):
        with torch.no_grad():
            model_fn(model, data)
        pdist_samples_dict = _sample_from_pdist(model)
        model_fn(model, data)
        loss = _compute_loss(model, pdist_samples_dict)
        model.zero_grad()
        return loss
    closure_fn = default_fn
    return closure_fn
