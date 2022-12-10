import os
import logging
from pathlib import Path
from tensorboardX import SummaryWriter

def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v

def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, str(k))
        else:
            flatted[str(k)] = v
    return flatted

def metric_to_str(metrics, sep='.'):
    flatted_metrics = flat_nested_json_dict(metrics, sep)
    metrics_str_list = []
    for k, v in flatted_metrics.items():
        if isinstance(v, float):
            metrics_str_list.append(f"{k}={v:.4}")
        elif isinstance(v, (list, tuple)):
            if v and isinstance(v[0], float):
                v_str = ', '.join([f"{e:.4}" for e in v])
                metrics_str_list.append(f"{k}=[{v_str}]")
            else:
                metrics_str_list.append(f"{k}={v}")
        else:
            metrics_str_list.append(f"{k}={v}")
    return ', '.join(metrics_str_list)

class Logger:
    """For simple log.
    generate 3 kinds of log: 
    1. simple log.txt, all metric dicts are flattened to produce
    readable results.
    2. TensorBoard scalars and images
    3. save images for visualization
    once it is initilized, all later usage will based on the initilized path
    """
    g_global_dir = None
    g_tsbd = None
    def __init__(self):
        pass

    @property
    def global_dir(self):
        return Logger.g_global_dir

    @global_dir.setter
    def global_dir(self, v):
        assert os.path.isdir(v), f"{v} is not a valid dir."
        Logger.g_global_dir=v
        logging.basicConfig(filename=Path(v, 'log.txt'))
        Logger.g_tsbd = SummaryWriter(v)
    
    @staticmethod
    def log_txt(s):
        if Logger.g_global_dir is None:
            print(s)
        else:
            print(s)
            logging.critical(s)

    @staticmethod
    def log_img(self, img, path):
        raise NotImplementedError
    
    @staticmethod
    def log_tsbd_scalor(k, v, epoch):
        Logger.g_tsbd.add_scalar(k, v, epoch)

    @staticmethod
    def log_tsbd_img(k, img, epoch):
        Logger.g_tsbd.add_image(k, img, epoch, dataformats='HWC')

    @staticmethod
    def log_metrics(metrics: dict, step):
        flatted_summarys = flat_nested_json_dict(metrics, "/")
        for k, v in flatted_summarys.items():
            if isinstance(v, (list, tuple)):
                if any([isinstance(e, str) for e in v]):
                    continue
                v_dict = {str(i): e for i, e in enumerate(v)}
                for k1, v1 in v_dict.items():
                    Logger.g_tsbd.add_scalar(k + "/" + k1, v1, step)
            else:
                if isinstance(v, str):
                    continue
                Logger.g_tsbd.add_scalar(k, v, step)
        log_str = metric_to_str(metrics)
        logging.critical(log_str)

if __name__ == "__main__":
    logger1 = Logger()
    logger1.global_dir = "/usr/app"
    logger2 = Logger()
    print(logger1.global_dir)
    print(logger2.global_dir)