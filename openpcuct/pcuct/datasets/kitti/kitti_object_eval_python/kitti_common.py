import re
import pathlib

from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common \
    import get_label_annos, filter_annos_low_score
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common \
    import get_image_index_str
from det3.ops import read_pkl, write_pkl, write_txt
from det3.dataloader.kittidata import KittiCalib

def save_eval_result(eval_result, dt_path):
    '''
    Save evaluation results.
    Args:
        eval_results (Tuple): (result_str, result_dict), which is the results of get_official_eval_result();
        dt_path (str): path to the directory container detection results.
    '''
    result_str, result_dict = eval_result
    save_dir = pathlib.Path(dt_path).absolute().parents[0]
    write_pkl(result_dict, save_dir/"eval_uct_result.pkl")
    write_txt([result_str], save_dir/"eval_uct_result.txt")

def get_uncertainty_anno(uncertainty_path):
    '''
    Args:
        uncertainty_path (str): path of an uncertainty pkl file
    Returns:
        pdist: it contains predictive distribution (BEV&3D) for each object.
            each dictionary contains:{
                "post_mean_bev": np.ndarray (5,)/(6,),
                "post_cov_bev": np.ndarray (5,5)/(6,6),
                "post_mean_3d": post_mean_3d (7,)/(8,),
                "post_cov_3d": post_cov_3d (7,7)/(8,8)}
    '''
    uncertainty = read_pkl(uncertainty_path)
    return uncertainty

def get_uncertainty_annos(uncertainty_folder, annos, image_ids=None):
    '''
    Append uncertainty in each anno.
    Args:
        uncertainty_folder (str): path to the directory containing uncertainty
            pkl files.
        annos (List[Dict])
        image_ids (List(str))
    Return:
        annos (List[Dict])
    '''
    if image_ids is None:
        filepaths = pathlib.Path(uncertainty_folder).glob('*.pkl')
        prog = re.compile(r'^\d{6}.pkl$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    uncertainty_folder = pathlib.Path(uncertainty_folder)
    for anno, idx in zip(annos, image_ids):
        image_idx = get_image_index_str(idx)
        label_filename = uncertainty_folder / (image_idx + '.pkl')
        uct = get_uncertainty_anno(label_filename)
        anno.update({"uncertainty": uct, "frameid": [image_idx]*len(uct)})
        err_msg = f"{len(uct)} {anno['name'].shape} {image_idx}"
        assert len(uct) == anno['name'].shape[0], err_msg
    return annos

def get_calibs(calib_folder, annos, image_ids=None, label_folder=None):
    '''
    Append calib in each anno.
    Args:
        calib_folder (str): path to the directory containing calib txt files.
        annos (List[Dict])
        image_ids (List(str))
        label_folder (str): path to the directory containing labels. The files
            in this folder will provide image_ids, if <image_ids> is None.
    Return:
        annos (List[Dict])
    '''
    if image_ids is None:
        assert label_folder is not None
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    calib_folder = pathlib.Path(calib_folder)
    for anno, idx in zip(annos, image_ids):
        image_idx = get_image_index_str(idx)
        calib_filename = calib_folder / (image_idx + '.txt')
        calib = KittiCalib(calib_filename).read_calib_file()
        anno.update({"calib": [calib]*len(anno['name'])})
    return annos