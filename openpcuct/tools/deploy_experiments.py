import os
import glob
import random
from det3.ops.ops import read_txt
# this line is to fix the mkl-service error
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import fire
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from det3.ops import write_txt

MACHINE = "LocalHost"
DEBUG = False

def run_cmd_list_single(cmd_list):
    for cmd in cmd_list:
        os.system(cmd)
    return

def run_cmd_list_parallel(cmd_list):
    with Pool(8) as p:
        r = list(tqdm(p.imap(os.system, cmd_list),
            total=len(cmd_list)))
    return

def _setup_wandb(exp_name):
    if DEBUG:
        return

def _log(**kwargs):
    if DEBUG:
        print(kwargs)
        return

def _save(file_path):
    if DEBUG:
        return

def _wandb_finish():
    if DEBUG:
        return

class Runner:
    '''
    Template Class
    '''
    def __init__(self, exp_name, param_dict, debug=False):
        '''
        Args:
            exp_name(str)
            param_dict(Dict)
        '''
        global DEBUG
        self.exp_name = exp_name
        self.param_dict = param_dict
        self.status = "Running"
        self.debug = debug
        DEBUG = debug

    def setup(self):
        '''
        Call this function before run().
        '''
        if self.debug:
            return
        _setup_wandb(self.exp_name)
        _log(exp_name=self.exp_name,
            status=self.status,
            machine=MACHINE,
            **self.param_dict)

    def run(self):
        raise NotImplementedError

    def collect(self):
        '''
        Call this function after run().
        '''
        self.status = "Finished"
        if self.debug:
            return
        _log(exp_name=self.exp_name, status=self.status)
        _wandb_finish()

class GenerateSpatialUncertainty(Runner):
    def run(self):
        cmd = "python3 gen_pred_uct.py "
        for k, v in self.param_dict.items():
            cmd += f"--{k}={v} "
        _log(cmd=cmd)
        run_cmd_list_single([cmd])

    @property
    def save_dir(self):
        if "is_label" in self.param_dict.keys():
            save_dir = (Path(self.param_dict["data_dir"])/"uncertainty").absolute()
        else:
            save_dir = (Path(self.param_dict["anno_dir"])/"../uncertainty").absolute()
        return str(save_dir)

    def collect(self):
        path = Path(self.save_dir)/f"__{self.exp_name}__"
        write_txt([""], path)
        super().collect()

class CalculateFisher(Runner):
    def run(self):
        set_kwargs = {k: v
            for k, v in self.param_dict.items()
            if "." in k}
        for k in set_kwargs.keys():
            self.param_dict.pop(k)
        cmd = "python3 calc_weight_uct.py "
        for k, v in self.param_dict.items():
            if k == "fix_random_seed":
                cmd = cmd + f"--{k} " if v else cmd
            else:
                cmd += f"--{k}={v} "
        for k, v in set_kwargs.items():
            cmd += f"--set {k} {v}"
        _log(cmd=cmd)
        run_cmd_list_single([cmd])

class GeneratePredictiveDistributions(Runner):
    def run(self):
        set_kwargs = {k: v
            for k, v in self.param_dict.items()
            if "." in k}
        for k in set_kwargs.keys():
            self.param_dict.pop(k)
        cmd = "python3 test.py --bool_uct --save_to_file "
        for k, v in self.param_dict.items():
            if k == "fix_random_seed":
                cmd = cmd + f"--{k} " if v else cmd
            else:
                cmd += f"--{k}={v} "
        for k, v in set_kwargs.items():
            cmd += f"--set {k} {v}"
        _log(cmd=cmd)
        run_cmd_list_single([cmd])

    def collect(self):
        self._save_logs()
        self._save_figures()
        super().collect()
    
    def _save_logs(self):
        if DEBUG:
            return
        root_dir = Path().absolute().parents[0]
        extra_tag = self.param_dict.get('extra_tag', 'default')
        cfg = self.param_dict['cfg']
        name = Path(cfg).name.split(".")[0]
        output_dir = root_dir/"output/kitti_models"/name
        output_dir = self._find_folder(str(output_dir),
            extra_tag)
        log_paths = glob.glob(str(Path(output_dir)/"**/log*.txt"), recursive=True)
        for log_path in log_paths:
            _save(str(Path(log_path).absolute()))
    
    def _save_figures(self):
        if DEBUG:
            return
        root_dir = Path().absolute().parents[0]
        extra_tag = self.param_dict.get('extra_tag', 'default')
        cfg = self.param_dict['cfg']
        name = Path(cfg).name.split(".")[0]
        output_dir = root_dir/"output/kitti_models"/name
        output_dir = self._find_folder(str(output_dir),
            extra_tag)
        output_dir = self._find_folder(str(output_dir),
            "vis-")
        zip_path = Path(output_dir)/'..'/f'{extra_tag}.tar.gz'
        cmd = f"tar -cvf {zip_path} {output_dir}"
        run_cmd_list_single([cmd])
        _save(str(zip_path))
    
    def _find_folder(self, root_dir, key):
        for itm in Path(root_dir).rglob("**/"):
            folder_name = str(itm).split("/")[-1]
            if key in folder_name:
                return Path(itm).absolute()

class GenerateDeterministicPredictions(Runner):
    def run(self):
        cmd = "python3 test.py --save_to_file "
        for k, v in self.param_dict.items():
            if k == "fix_random_seed":
                cmd = cmd + f"--{k} " if v else cmd
            else:
                cmd += f"--{k}={v} "
        _log(cmd=cmd)
        run_cmd_list_single([cmd])

class EvaluatePredictiveDistributions(Runner):
    def run(self):
        data_dir = self.param_dict["data_dir"]
        pred_dir = self.param_dict["pred_dir"]
        root_dir = Path().absolute().parents[0]
        eval_dir = root_dir/"pcuct/datasets/kitti/kitti_object_eval_python"
        cmd = f"cd {eval_dir} && "
        cmd+= f"python evaluate.py evaluate_uct "
        cmd+= f"--label_path={Path(data_dir)/'label_2'} "
        cmd+= f"--label_uct_path={Path(data_dir)/'uncertainty'} "
        cmd+= f"--result_path={Path(pred_dir)/'data'} "
        cmd+= f"--result_uct_path={Path(pred_dir)/'uncertainty'} "
        cmd+= f"--calib_path={Path(data_dir)/'calib'} "
        cmd+= f"--label_split_file={Path(data_dir).absolute().parent/'ImageSets/val.txt'} "
        cmd+= f"--current_class=0,1,2 "
        _log(cmd=cmd)
        run_cmd_list_single([cmd])
    
    def collect(self):
        self._log_result_dict()
        super().collect()
    
    def _log_result_dict(self):
        pred_dir = self.param_dict["pred_dir"]
        result_pkl_path = Path(pred_dir)/"eval_uct_result.pkl"
        result_txt_path = Path(pred_dir)/"eval_uct_result.txt"
        _save(str(result_pkl_path))
        _save(str(result_txt_path))

class CreateTuneInfoPKL(Runner):
    def run(self):
        split_path = self.param_dict["split_path"]
        num_data = self.param_dict["num_data"]
        root_dir = Path().absolute().parents[0]
        # create a subset of split
        split = read_txt(split_path)
        random.shuffle(split)
        sub_split = split[:num_data]
        sub_split.sort(key=lambda k: int(k))
        save_path_split = Path(split_path).parent / "tune.txt"
        write_txt(sub_split, str(save_path_split))
        # generate an info pkl
        cmd = f"cd {str(root_dir)} && "
        cmd+= "python -m pcdet.datasets.kitti.kitti_dataset "
        cmd+= "create_kitti_tune_infos "
        cmd+= "tools/cfgs/dataset_configs/kitti_dataset.yaml"
        _log(cmd=cmd)
        run_cmd_list_single([cmd])

    def collect(self):
        self._save_result()
        super().collect()
    
    def _save_result(self):
        split_path = self.param_dict["split_path"]
        save_path_split = Path(split_path).parent / "tune.txt"
        _save(str(save_path_split))

class ExperimentRunner:
    '''
    - generate spatial uncertainty with generative model
    - generate deterministic predictions with SOTA detectors
    - generate predictive distributions with laplace approximation
    - calculate fisher information matrices
    - evaluate predictive distributions
    '''
    def generate_ground_truth_spatial_uncertainty(
        self,
        exp_name,
        data_dir,
        debug=False):
        runner = GenerateSpatialUncertainty(exp_name,
            param_dict={
                "data_dir": data_dir,
                "anno_dir": str(Path(data_dir)/"label_2")},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()
    
    def generate_estimation_spatial_uncertainty(
        self,
        exp_name,
        data_dir,
        anno_dir,
        debug=False):
        runner = GenerateSpatialUncertainty(exp_name,
            param_dict={
                "data_dir": data_dir,
                "anno_dir": anno_dir},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()
    
    def calculate_fisher(
        self,
        exp_name,
        cfg,
        ckpt,
        fix_random_seed=False,
        debug=False, **kwargs):
        runner = CalculateFisher(exp_name,
            param_dict={
                "cfg": cfg,
                "ckpt": ckpt,
                "fix_random_seed": fix_random_seed,
                "batch_size": 1, **kwargs},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()

    def generate_predictive_distributions(
        self,
        exp_name,
        cfg,
        num_MC_samples,
        weight_dist_path,
        alpha,
        scaler_weight_prior,
        fix_random_seed=False,
        debug=False, **kwargs):
        runner = GeneratePredictiveDistributions(exp_name,
            param_dict={
            "cfg": cfg,
            "num_MC_samples": num_MC_samples,
            "weight_dist_path": weight_dist_path,
            "alpha": alpha,
            "scaler_weight_prior": scaler_weight_prior,
            "fix_random_seed": fix_random_seed,
            **kwargs},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()

    def generate_deterministic_predictions(
        self,
        exp_name,
        cfg,
        fix_random_seed=False,
        debug=False, **kwargs):
        runner = GenerateDeterministicPredictions(exp_name,
            param_dict={
            "cfg": cfg,
            "fix_random_seed": fix_random_seed,
            **kwargs},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()
    
    def evaluate_predictive_distributions(
        self,
        exp_name,
        data_dir,
        pred_dir,
        debug=False):
        runner = EvaluatePredictiveDistributions(exp_name,
            param_dict={
            "data_dir": data_dir,
            "pred_dir": pred_dir},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()

    def create_tune_info_pkl(
        self,
        exp_name,
        split_path,
        num_data,
        debug=False):
        runner = CreateTuneInfoPKL(exp_name,
            param_dict={
                "split_path": split_path,
                "num_data": num_data},
            debug=debug)
        runner.setup()
        runner.run()
        runner.collect()

if __name__ == "__main__":
    fire.Fire(ExperimentRunner)
