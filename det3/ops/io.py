'''
 File Created: Sat Feb 29 2020

'''
import json
import pickle
import numpy as np
from PIL import Image

def read_txt_(path:str):
    with open(path, 'r') as f:
        l = f.readlines()
    l = [itm.rstrip() for itm in l]
    return l

def write_txt_(obj:list, path:str):
    s = "\n".join(obj)
    with open(path, 'w+') as f:
        f.write(s)

def read_npy_(path:str):
    surf = path.split(".")[-1]
    assert surf == "npy"
    p = np.load(path)
    assert not np.isnan(np.sum(p))
    return p

def write_npy_(obj:np.ndarray, path:str):
    surf = path.split(".")[-1]
    assert surf == "npy"
    np.save(path, obj)

def read_pcd_(path:str):
    import open3d
    surf = path.split(".")[-1]
    assert surf == "pcd"
    pcd = open3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)

def write_pcd_(obj:np.ndarray, path:str):
    import open3d
    surf = path.split(".")[-1]
    assert surf == "pcd"
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(obj)
    open3d.io.write_point_cloud(path, pcd)

def read_bin_(path:str, dtype):
    return np.fromfile(path, dtype=dtype)
    
def write_bin_(obj:np.ndarray, path:str):
    with open(path, 'wb') as f:
            obj.tofile(f)

def read_img_(path:str):
    return np.array(Image.open(path, 'r'))

def write_img_(obj:np.ndarray, path:str):
    Image.fromarray(obj).save(path)

def read_pkl_(path:str):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def write_pkl_(obj, path:str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def read_json_(path:str):
    with open(path, encoding='utf-8') as f:
        res = f.read()
        result = json.loads(res)
    return result
