import torch
import time
import sys
import numpy as np

def printr(obj, pref=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            printr(v, pref=f"{pref}{k}:")
        return
    elif isinstance(obj, list):
        for i, itm in enumerate(obj):
            printr(itm, pref=f"{pref}{i}-")
        return
    elif type(obj) in [int, float, str, bool, type(None), tuple]:
        print(f"{pref} {obj}")
    elif isinstance(obj, np.ndarray):
        print(f"{pref} {obj.shape}")
    elif isinstance(obj, torch.Tensor):
        print(f"{pref} {obj.shape}")
    else:
        try:
            print(f"{pref} {obj.shape}")
        except:
            wrn_msg = f"Unrecognized type: {type(obj)}"
            print(wrn_msg)
            print(obj)
    
class Debug:
    def __init__(self, exit=True, s='Debug'):
        self.s = s
        self.exit = exit

    def __enter__(self):
        print(f"========= {self.s} START =========")

    def __exit__(self, type, value, traceback):
        if traceback is not None:
            print(f"========= {self.s} END =========")
            return
        if self.exit:
            sys.exit(f"========= {self.s} END =========")
        else:
            print(f"========= {self.s} END =========")

class Timer:
    def __init__(self, s='Timer'):
        self.s = s
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        print(f"========= {self.s} {time.time()-self.start_time:.3f} =========")