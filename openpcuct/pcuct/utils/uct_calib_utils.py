from det3.ops.ops import write_pkl
import torch
import numpy as np
from pathlib import Path
from scipy.stats import norm, multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm

from pcuct.datasets.kitti.kitti_object_eval_python import kitti_common
from pcuct.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def match_gt_and_det_multivariate_regression(
    data_dir,
    anno_dir,
    uct_dir,
    label_split_file,
    dims=[0,1],
    threshold=0.7):
    '''
    Read ground-truth and detections from <data_dir> and <anno_dir>, respectively.
    Match them according to rotatedBEV IoU and return a bijection.
    Args:
        data_dir
        anno_dir
        uct_dir
        label_split_file
        dims (int default=[0,1]): the dimension indices in the output space
        threshold (float default=0.7): threshold for treating a detection as TP.
    Returns:
        covs (torch.FloatTensor (N,)): N is the length of the found gt-det pairs
        preds (torch.FloatTensor (N,))
        labels (torch.FloatTensor (N,))
    '''
    def _anno_to_box(anno, idx):
        loc = anno["location"][idx].flatten()
        dims = anno["dimensions"][idx].flatten()
        rots = np.array(anno["rotation_y"][idx])
        bbox3d_Fcam = np.concatenate([loc, dims, rots[..., np.newaxis]])
        calib = anno["calib"][idx]
        x, y, z, l, h, w, ry = bbox3d_Fcam.reshape(-1)
        bottom_Fcam = np.array([[x, y, z]])
        bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
        bbox3d = np.array([
            bottom_Flidar[0,0],
            bottom_Flidar[0,1],
            bottom_Flidar[0,2]+h/2.0, # center
            l, w, h,
            np.pi/2-ry]).reshape(1,7)
        return bbox3d

    def _align_rot_fn(pred, label, dims):
        idx = dims.index(6)
        rtn = pred.copy()
        rtn[:, idx] -= np.pi * np.around((pred[:, idx] - label[:, idx]) / np.pi)
        return rtn

    dims = [int(itm) for itm in dims]
    assert all([itm>=0 for itm in dims]), "dim should be a non-negative int."
    is_rot = False
    if 6 in dims:
        wrn_msg = "=>Warning: dim==6 indicates the rotation dimension."
        wrn_msg += "It will automatically align pred and label "
        wrn_msg += "to eliminate the pi periods."
        print(wrn_msg)
        is_rot = True

    dt_annos = kitti_common.get_label_annos(anno_dir)
    dt_annos = kitti_common.get_uncertainty_annos(
        uct_dir, dt_annos)
    dt_annos = kitti_common.get_calibs(
        str(Path(data_dir)/"calib"), dt_annos, label_folder=anno_dir)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti_common.get_label_annos(
        str(Path(data_dir) / "label_2"), val_image_ids)
    gt_annos = kitti_common.get_calibs(
        str(Path(data_dir)/"calib"), gt_annos, image_ids=val_image_ids)
    assert len(gt_annos) == len(dt_annos)

    num_examples = len(gt_annos)
    kitti_eval.get_split_parts(num_examples, num_part=50)
    rets = kitti_eval.calculate_iou_partly(dt_annos, gt_annos, metric=1, num_parts=50)
    overlaps, _, _, _ = rets
    covs, preds, labels = [], [], []
    for dt_anno, gt_anno, overlap in zip(dt_annos, gt_annos, overlaps):
        len_dt = len(dt_anno["name"])
        for dt_i in range(len_dt):
            overlap_ = overlap[dt_i]
            gt_i = np.argmax(overlap_)
            if overlap_[gt_i] > threshold:
                cov = dt_anno['uncertainty'][dt_i]["post_cov_3d"]
                assert cov.shape[0] == cov.shape[1] == 7
                cov = np.concatenate([
                    cov[i, dims].reshape(1, 1, -1) for i in dims], axis=1)
                # cov_v = np.array([cov[dim_i, dim_i] for dim_i in dims])
                # cov = np.diag(cov_v).reshape(1, len(dims), len(dims))
                dt_box_Flidar = _anno_to_box(dt_anno, dt_i).reshape(-1)
                gt_box_Flidar = _anno_to_box(gt_anno, gt_i).reshape(-1)
                pred = dt_box_Flidar[dims].reshape(1, -1)
                label = gt_box_Flidar[dims].reshape(1, -1)
                pred = _align_rot_fn(pred, label, dims) if is_rot else pred
                covs.append(cov)
                preds.append(pred)
                labels.append(label)
    covs = np.concatenate(covs, axis=0)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    return torch.from_numpy(covs).float(), \
           torch.from_numpy(preds).float(), \
           torch.from_numpy(labels).float()

def make_model_diagrams_multivariate_regression(covs, preds, labels, n_bins=10):
    """
    Args:
        covs (np.ndarray (N, D, D)): variances of predictions
        preds (np.ndarray (N, D)): predictions
        labels (np.ndarray (N, D)): ground-truth labels
        n_bins (int)
    Returns:
        ece (float): expected calibrated error
    """
    n_dims = labels.shape[-1]
    width = 1/n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    cdfs = np.array([
        multivariate_normal.cdf(label, mean=pred, cov=cov)
        for cov, pred, label in tqdm(zip(covs, preds, labels), total=labels.shape[0])])
    cdfs *= 2**(n_dims -1)
    emp_freqs = np.array([
        (cdfs < bin_center).mean()
        for bin_center in bin_centers])
    # compute ece
    bin_freqs = np.array([
        emp_freqs[i] - emp_freqs[i-1] if i!= 0 else emp_freqs[i]
        for i in range(n_bins)])
    ece = bin_freqs * np.abs(emp_freqs - bin_centers)
    ece = ece.sum()

    plt.figure(0, figsize=(8, 8))
    plt.bar(bin_centers, emp_freqs, width=width, alpha=0.1, ec='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
    plt.text(0.2, 0.85, "ECE: {:.2f}".format(ece), ha="center", va="center", size=20, weight = 'bold', bbox=bbox_props)

    plt.title("Reliability Diagram", size=20)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return ece

def match_gt_and_det_regression(
    data_dir,
    anno_dir,
    uct_dir,
    label_split_file,
    dim=0,
    threshold=0.7):
    '''
    Read ground-truth and detections from <data_dir> and <anno_dir>, respectively.
    Match them according to rotatedBEV IoU and return a bijection.
    Args:
        data_dir
        anno_dir
        uct_dir
        label_split_file
        dim (int default=0): the dimension index in the output space
        threshold (float default=0.7): threshold for treating a detection as TP.
    Returns:
        covs (torch.FloatTensor (N,)): N is the length of the found gt-det pairs
        preds (torch.FloatTensor (N,))
        labels (torch.FloatTensor (N,))
    '''
    def _anno_to_box(anno, idx):
        loc = anno["location"][idx].flatten()
        dims = anno["dimensions"][idx].flatten()
        rots = np.array(anno["rotation_y"][idx])
        bbox3d_Fcam = np.concatenate([loc, dims, rots[..., np.newaxis]])
        calib = anno["calib"][idx]
        x, y, z, l, h, w, ry = bbox3d_Fcam.reshape(-1)
        bottom_Fcam = np.array([[x, y, z]])
        bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
        bbox3d = np.array([
            bottom_Flidar[0,0],
            bottom_Flidar[0,1],
            bottom_Flidar[0,2]+h/2.0, # center
            l, w, h,
            np.pi/2-ry]).reshape(1,7)
        return bbox3d

    assert dim >= 0, "dim should be a non-negative int."
    is_rot = False
    if dim == 6:
        wrn_msg = "=>Warning: dim==6 indicates the rotation dimension."
        wrn_msg += "It will automatically align pred and label "
        wrn_msg += "to eliminate the pi periods."
        print(wrn_msg)
        is_rot = True
        _align_rot_fn = lambda pred, label: \
            pred - np.pi * np.around((pred - label) / np.pi)
    dt_annos = kitti_common.get_label_annos(anno_dir)
    dt_annos = kitti_common.get_uncertainty_annos(
        uct_dir, dt_annos)
    dt_annos = kitti_common.get_calibs(
        str(Path(data_dir)/"calib"), dt_annos, label_folder=anno_dir)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti_common.get_label_annos(
        str(Path(data_dir) / "label_2"), val_image_ids)
    gt_annos = kitti_common.get_calibs(
        str(Path(data_dir)/"calib"), gt_annos, image_ids=val_image_ids)
    assert len(gt_annos) == len(dt_annos)

    num_examples = len(gt_annos)
    kitti_eval.get_split_parts(num_examples, num_part=50)
    rets = kitti_eval.calculate_iou_partly(dt_annos, gt_annos, metric=1, num_parts=50)
    overlaps, _, _, _ = rets
    covs, preds, labels = [], [], []
    for dt_anno, gt_anno, overlap in zip(dt_annos, gt_annos, overlaps):
        len_dt = len(dt_anno["name"])
        for dt_i in range(len_dt):
            overlap_ = overlap[dt_i]
            gt_i = np.argmax(overlap_)
            if overlap_[gt_i] > threshold:
                cov = dt_anno['uncertainty'][dt_i]["post_cov_3d"]
                assert cov.shape[0] == cov.shape[1] == 7
                dim = int(dim)
                cov = cov[dim, dim]
                dt_box_Flidar = _anno_to_box(dt_anno, dt_i).reshape(-1)
                gt_box_Flidar = _anno_to_box(gt_anno, gt_i).reshape(-1)
                pred = dt_box_Flidar[dim]
                label = gt_box_Flidar[dim]
                pred = _align_rot_fn(pred, label) if is_rot else pred
                covs.append(cov)
                preds.append(pred)
                labels.append(label)

    return torch.Tensor(covs).float(), \
           torch.Tensor(preds).float(), \
           torch.Tensor(labels).float()

def make_model_diagrams_regression(covs, preds, labels, n_bins=10, plot_data_path=None):
    """
    Args:
        covs (np.ndarray (N, )): variances of predictions
        preds (np.ndarray (N, )): predictions
        labels (np.ndarray (N, )): ground-truth labels
        n_bins (int)
    Returns:
        ece (float): expected calibrated error
    """
    width = 1/n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    cdfs = np.array([
        norm.cdf(label, loc=pred, scale=cov**0.5)
        for cov, pred, label in zip(covs, preds, labels)])
    emp_freqs = np.array([
        (cdfs < bin_center).mean()
        for bin_center in bin_centers])
    # compute ece
    bin_freqs = np.array([
        emp_freqs[i] - emp_freqs[i-1] if i!= 0 else emp_freqs[i]
        for i in range(n_bins)])
    ece = bin_freqs * np.abs(emp_freqs - bin_centers)
    ece = ece.sum()

    if plot_data_path is not None:
        write_pkl(dict(x_vals=bin_centers,
                       y_vals=emp_freqs,
                       ece=ece), plot_data_path)

    plt.figure(0, figsize=(8, 8))
    plt.bar(bin_centers, emp_freqs, width=width, alpha=0.1, ec='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
    plt.text(0.2, 0.85, "ECE: {:.2f}".format(ece), ha="center", va="center", size=20, weight = 'bold', bbox=bbox_props)

    plt.title("Reliability Diagram", size=20)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return ece

def match_gt_and_det_classification(
    data_dir,
    anno_dir,
    label_split_file,
    threshold=0.7):
    '''
    Read ground-truth and detections from <data_dir> and <anno_dir>, respectively.
    Match them according to rotatedBEV IoU and return a bijection.
    Args:
        data_dir
        anno_dir
        label_split_file
        threshold (float default=0.7): threshold for treating a detection as TP.
    Returns:
        scores (torch.FloatTensor (N,)): N is the length of the found gt-det pairs
        preds (torch.LongTensor (N,))
        labels (torch.LongTensor (N,))
    '''
    dt_annos = kitti_common.get_label_annos(anno_dir)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti_common.get_label_annos(
        str(Path(data_dir) / "label_2"), val_image_ids)
    assert len(gt_annos) == len(dt_annos)

    num_examples = len(gt_annos)
    kitti_eval.get_split_parts(num_examples, num_part=50)
    rets = kitti_eval.calculate_iou_partly(dt_annos, gt_annos, metric=1, num_parts=50)
    overlaps, _, _, _ = rets
    scores, preds, labels = [], [], []
    name_to_idx = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 0,
        'Person_sitting': 1,
        'Truck': 0,
        "Background": 3}
    for dt_anno, gt_anno, overlap in zip(dt_annos, gt_annos, overlaps):
        len_dt = len(dt_anno["name"])
        for dt_i in range(len_dt):
            overlap_ = overlap[dt_i]
            gt_i = np.argmax(overlap_)
            score = dt_anno['score'][dt_i]
            pred = dt_anno['name'][dt_i]
            label = gt_anno['name'][gt_i] \
                if overlap_[gt_i] > threshold else "Background"
            if label in ["DontCare", "Misc", "Tram"]:
                continue
            scores.append(score)
            preds.append(name_to_idx[pred])
            labels.append(name_to_idx[label])

    return torch.Tensor(scores).float(), \
           torch.Tensor(preds).int(), \
           torch.Tensor(labels).int()

'''The bellow functions are adapted from https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71'''

def calculate_ece_classification(scores, preds, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Nobody: modification
    # softmaxes = F.softmax(logits, dim=1)
    # confidences, predictions = torch.max(softmaxes, 1)
    confidences, predictions = scores, preds

    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=predictions.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def make_model_diagrams_classification(scores, preds, labels, n_bins=10, plot_data_path=None):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    # Nobody: modification
    # softmaxes = torch.nn.functional.softmax(outputs, 1)
    # confidences, predictions = softmaxes.max(1)
    confidences, predictions = scores, preds
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
     
    plt.figure(0, figsize=(8, 8))
    confs = plt.bar(bin_centers, bin_corrects, width=width, alpha=0.1, ec='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    # Nobody: modification
    # bin_corrects = np.nan_to_num(bin_corrects)
    # bin_scores = np.nan_to_num(bin_scores)
    # gap = np.array(bin_scores - bin_corrects)
    # gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    # plt.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

    # Nobody: modification
    # ece = calculate_ece(outputs, labels)
    ece = calculate_ece_classification(scores, preds, labels)
    if plot_data_path is not None:
        write_pkl(dict(x_vals=bin_centers,
                       y_vals=bin_corrects,
                       ece=ece), plot_data_path)

    # Clean up
    bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
    plt.text(0.2, 0.85, "ECE: {:.2f}".format(ece), ha="center", va="center", size=20, weight = 'bold', bbox=bbox_props)

    plt.title("Reliability Diagram", size=20)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return ece

'''The functions above are adapted from https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71'''