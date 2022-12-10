import fire
from pcdet.datasets.kitti.kitti_object_eval_python.evaluate \
    import _read_imageset_file

import kitti_common as kitti
from eval import get_official_eval_result

def evaluate_uct(
            label_path,
            label_uct_path,
            result_path,
            result_uct_path,
            calib_path,
            label_split_file,
            current_class=0,
            coco=False,
            score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    dt_annos = kitti.get_uncertainty_annos(result_uct_path, dt_annos)
    dt_annos = kitti.get_calibs(calib_path, dt_annos, label_folder=result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    gt_annos = kitti.get_uncertainty_annos(label_uct_path, gt_annos, val_image_ids)
    gt_annos = kitti.get_calibs(calib_path, gt_annos, image_ids=val_image_ids)
    if coco:
        err_msg = "COCO is not supported in evaluate_uct."
        raise NotImplementedError(err_msg)
    else:
        rtn = get_official_eval_result(gt_annos, dt_annos, current_class, bool_uct=True)
        kitti.save_eval_result(rtn, result_path)
        return rtn

if __name__ == '__main__':
    fire.Fire()