import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from det3.ops import read_pkl
from matplotlib import pyplot as plt
from tqdm import tqdm

def run_cmd_list_single(cmd_list):
    for cmd in tqdm(cmd_list):
        os.system(cmd)
    return

name2epoch = {
    "pointpillar": "7728",
    "second": "7862",
    "pointrcnn": "7870"
}

def plot_calibration_plots(save_dir, data_dir, experiment_data_dir):
    def _assemble_cmds(tag):
        threshold = 0.1
        if "mcdropout" in tag:
            name, mode, seed, bnn_mode, _ = tag.split("-")
            sample_method = "heuristic"
        else:
            name, mode, seed, bnn_mode1, bnn_mode2, _ = tag.split("-")
            bnn_mode = f"{bnn_mode1}_{bnn_mode2}"
            sample_method = "heuristic"
        seed = seed[4:]
        epoch = name2epoch[name]
        anno_dir = Path(experiment_data_dir)
        anno_dir = anno_dir/tag
        anno_dir = anno_dir/f"eval/epoch_{epoch}/val/default/final_result/data"
        label_split_file = Path(data_dir)/"../ImageSets/val.txt"

        title = f"{name}-{epoch}-{mode}-{bnn_mode}-{seed}-cls"
        cmds = []
        if not (Path(save_dir)/f"{title}.pkl").exists():
            cmd0 = "python3 analyze_results.py draw_classification_calibration_plot "
            kwargs0 = dict(data_dir=str(data_dir),
                        anno_dir=str(anno_dir),
                        label_split_file=str(label_split_file),
                        save_path=str(Path(save_dir)/f"{title}.png"),
                        plot_data_path=str(Path(save_dir)/f"{title}.pkl"),
                        n_bins=10,
                        title=title, threshold=threshold)
            for k, v in kwargs0.items():
                cmd0 += f"--{k}={v} "
            cmds += [cmd0]

        dims = range(7)
        dim_names = ["xc", "yc", "zc", "l", "w", "h", "ry"]
        for dim, dim_name in zip(dims, dim_names):
            title = f"{name}-{epoch}-{mode}-{bnn_mode}-{seed}-{dim_name}"
            if not (Path(save_dir)/f"{title}.pkl").exists():
                kwargs1 = dict(data_dir=str(data_dir),
                    anno_dir=str(anno_dir),
                    uct_dir=str(anno_dir/"../uncertainty"),
                    label_split_file=str(label_split_file),
                    save_path=str(Path(save_dir)/f"{title}.png"),
                    plot_data_path=str(Path(save_dir)/f"{title}.pkl"),
                    n_bins=50,
                    dim=dim, title=title, threshold=threshold)
                cmd1 = "python3 analyze_results.py draw_regression_calibration_plot "
                for k, v in kwargs1.items():
                    cmd1 += f"--{k}={v} "
                cmds += [cmd1]
        return cmds

    tags = glob.glob(str(Path(experiment_data_dir)/"*-gen_pred_dist"))
    tags = [Path(itm).name for itm in tags]
    cmd_list = []
    for tag in tags:
        cmds = _assemble_cmds(tag)
        cmd_list += cmds
    run_cmd_list_single(cmd_list)

    # get all pkls
    data = {}
    for pkl_path in glob.glob(str(Path(save_dir)/"*.pkl")):
        pkl = read_pkl(pkl_path)
        if "deterministic" in pkl_path:
            continue
        name, _, mode, bnn_mode, _, dim_name = Path(pkl_path).name[:-4].split("-")
        data.update({f"{name}-{mode}-{bnn_mode}-{dim_name}":
            dict(x_vals=[float(itm) for itm in pkl["x_vals"]],
                 y_vals=[float(itm) if not np.isnan(itm) \
                     else 0.0 for itm in pkl["y_vals"]],
                 ece=pkl["ece"])})
    # plot
    modify_name={
        "pointpillar": "PP",
        "second": "SC",
        "pointrcnn": "PR"
    }
    modify_bnn_mode = {
        "full_net": "full",
        "last_layer": "LL",
        "last_module": "LM",
        "mcdropout": "MC",
    }
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 13,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })

    for name in ["pointpillar", "second", "pointrcnn"]:
        bnn_modes = ["mcdropout", "last_layer", "full_net"] \
            if name != "pointrcnn" \
            else ["mcdropout", "last_layer", "last_module", "full_net"]
        for bnn_mode in bnn_modes:
            modes = ["standard", "empirical"] \
                if bnn_mode != "mcdropout" else ["standard"]
            for mode in modes:
                plt.figure(figsize=(5, 5))
                ax = plt.subplot(1,1,1)
                for dim_name in ["cls", "xc", "yc", "zc", "l", "w", "h", "ry"]:
                    data_ = data[f"{name}-{mode}-{bnn_mode}-{dim_name}"]
                    ax.plot([0.0]+data_["x_vals"]+[1.0], [0.0]+data_["y_vals"]+[1.0], label=dim_name, linewidth=3)
                    ax.legend()
                plt.xlim(0,1)
                plt.ylim(0,1)
                plt.xlabel("Predicted confidence")
                plt.ylabel("Empirical confidence")
                plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=3)
                title = f"{modify_name[name]}-{modify_bnn_mode[bnn_mode]}({mode})"
                plt.title(title)
                plt.savefig(title+".pdf", dpi=200, bbox_inches="tight")


def print_ece_table(result_data_dir):
    def _get_val_from_data(data, name, mode, bnn_mode, seed, dim_name, key):
        try:
            return data[f"{name}-{mode}-{bnn_mode}-{seed}-{dim_name}"][key]
        except:
            return None

    def _compute_ece_statistics(
        data,
        name,
        mode,
        bnn_mode,
        seeds,
        return_err_seeds=False):
        err_seeds, ece_vals_dict = [], {}
        for dim_name in ["cls", "xc", "yc", "zc", "l", "w", "h", "ry"]:
            ece_vals_ = [
                _get_val_from_data(data, name, mode,
                    bnn_mode, seed, dim_name, "ece")
                for seed in seeds]
            ece_vals = []
            for ece_val, seed in zip(ece_vals_, seeds):
                if ece_val is not None:
                    ece_vals.append(ece_val)
                else:
                    err_seeds.append(seed)
            ece_vals_dict[dim_name] = ece_vals
        err_seeds = list(set(err_seeds))
        for dim_name in ["cls", "xc", "yc", "zc", "l", "w", "h", "ry"]:
            assert None not in ece_vals_dict[dim_name]
            err_msg = f"{name}-{mode}-{bnn_mode}-{seed}-{dim_name} error."
            assert len(ece_vals_dict[dim_name]) \
                == len(seeds) - len(err_seeds), err_msg

        ece_vals_dict["reg"] = np.array([
            ece_vals_dict[dim_name]
            for dim_name in ["xc", "yc", "zc", "l", "w", "h", "ry"]]
            ).sum(axis=0)
        ece_vals_dict["reg"] = ece_vals_dict["reg"].flatten()
        cls_stat = (np.mean(ece_vals_dict["cls"]), np.std(ece_vals_dict["cls"]))
        reg_stat = (np.mean(ece_vals_dict["reg"]), np.std(ece_vals_dict["reg"]))
        if not return_err_seeds:
            return {"classification": cls_stat, "regression": reg_stat}
        else:
            return {"classification": cls_stat, "regression": reg_stat}, err_seeds

    data = {}
    for pkl_path in glob.glob(str(Path(save_dir)/"*.pkl")):
        pkl = read_pkl(pkl_path)
        if "deterministic" in pkl_path:
            continue
        name, _, mode, bnn_mode, seed, dim_name = Path(pkl_path).name[:-4].split("-")
        data.update({f"{name}-{mode}-{bnn_mode}-{seed}-{dim_name}":
            dict(x_vals=[float(itm) for itm in pkl["x_vals"]],
                 y_vals=[float(itm) if not np.isnan(itm) \
                     else 0.0 for itm in pkl["y_vals"]],
                 ece=pkl["ece"])})

    table_dict = {}
    err_seeds_dict = {}
    bnn_modes_dict = {
        "pointpillar": ["last_layer", "full_net"],
        "second": ["last_layer", "full_net"],
        "pointrcnn": ["last_layer", "last_module", "full_net"],
    }
    # deterministic
    dtm_dict = {
        "pointpillar": dict(
            classification=(0.115, None),
            regression=(None, None)),
        "second": dict(
            classification=(0.100, None),
            regression=(None, None)),
        "pointrcnn": dict(
            classification=(0.113, None),
            regression=(None, None)),
    }
    # mcdropout
    mcdropout_dict = {}
    for name in ["pointpillar", "second", "pointrcnn"]:
        res_dict, err_seeds = _compute_ece_statistics(
                data, name, "standard", "mcdropout",
                seeds=[666],
                return_err_seeds=True)
        mcdropout_dict.update({name: res_dict})
        if len(err_seeds)> 0:
            err_seeds_dict.update({f"{name}-standard-mcdropout": err_seeds})
    # LA
    laplace_dict = {}
    for name in ["pointpillar", "second", "pointrcnn"]:
        for mode in ["standard", "empirical"]:
            for bnn_mode in bnn_modes_dict[name]:
                res_dict, err_seeds = _compute_ece_statistics(
                        data, name, mode, bnn_mode,
                        seeds=[666],
                        return_err_seeds=True)
                laplace_dict.update({f"{name}-{mode}-{bnn_mode}": res_dict})
                if len(err_seeds) > 0:
                    err_seeds_dict.update({f"{name}-{mode}-{bnn_mode}": err_seeds})

    for name in ["pointpillar", "second", "pointrcnn"]:
        table_dict[name] = {
            "deterministic": dtm_dict[name],
            "mcdropout": mcdropout_dict[name]}
    for name in ["pointpillar", "second", "pointrcnn"]:
        for mode in ["standard", "empirical"]:
            for bnn_mode in bnn_modes_dict[name]:
                table_dict[name].update({f"{bnn_mode}({mode})": \
                            laplace_dict[f"{name}-{mode}-{bnn_mode}"]})
    modify_name={
        "pointpillar": "PP",
        "second": "SC",
        "pointrcnn": "PR"
    }
    modify_bnnmode = {
        "deterministic": "DT",
        "mcdropout": "MC",
        "last_layer": "LL",
        "last_module": "LM",
        "full_net": "FU"
    }
    modify_mode = {
        "empirical": "emp.",
        "standard": "std."
    }
    plt_data_dict = {
        "classification-mean": {},
        "classification-std": {},
        "regression-mean": {},
        "regression-std": {},
    }
    for name in ["pointpillar", "second", "pointrcnn"]:
        for bnn_mode in ["deterministic", "mcdropout"]:
            tag = f"{modify_name[name]}-{modify_bnnmode[bnn_mode]}"
            cls_stats = table_dict[name][bnn_mode]["classification"]
            reg_stats = table_dict[name][bnn_mode]["regression"]
            plt_data_dict["classification-mean"][tag] = cls_stats[0]
            plt_data_dict["classification-std"][tag] = cls_stats[1]
            plt_data_dict["regression-mean"][tag] = reg_stats[0]
            plt_data_dict["regression-std"][tag] = reg_stats[1]
        for mode in ["empirical", "standard"]:
            for bnn_mode in ["last_layer", "last_module", "full_net"]:
                if name != "pointrcnn" and bnn_mode == "last_module":
                    continue
                tag = f"{modify_name[name]}-{modify_bnnmode[bnn_mode]}({modify_mode[mode]})"
                cls_stats = table_dict[name][f"{bnn_mode}({mode})"]["classification"]
                reg_stats = table_dict[name][f"{bnn_mode}({mode})"]["regression"]
                plt_data_dict["classification-mean"][tag] = cls_stats[0]
                plt_data_dict["classification-std"][tag] = cls_stats[1]
                plt_data_dict["regression-mean"][tag] = reg_stats[0]
                plt_data_dict["regression-std"][tag] = reg_stats[1]
    df_dict = {}
    for dim_name in ["classification", "regression"]:
        for stat_name in ["mean", "std"]:
            vals, keys = [], []
            for k, v in plt_data_dict[f"{dim_name}-{stat_name}"].items():
                keys.append(k)
                vals.append(v)
            df_dict[f"{dim_name}-{stat_name}"] = pd.Series(vals, index=keys)

    df = pd.DataFrame(df_dict)
    print(df)
    writer = pd.ExcelWriter(str(Path(result_data_dir)/"ECE_table.xlsx"), engine='xlsxwriter')
    df.to_excel(writer, sheet_name="expected calibration error")
    writer.save()
    print(f'Save ECE_table in {str(Path(result_data_dir)/"ECE_table.xlsx")}')
    print(f"Error seeds are:")
    for k, v in err_seeds_dict.items():
        print(k, v)

def plot_deterministic_calibration_plots(save_dir, data_dir, experiment_data_dir):
    def _assemble_cmds(name, epoch):
        threshold = 0.1
        anno_dir = Path(experiment_data_dir)
        anno_dir = anno_dir/f"{name}-deterministic-seed666-gen_det_pred"
        anno_dir = anno_dir/f"eval/epoch_{epoch}/val/default/final_result/data"
        label_split_file = Path(data_dir)/"../ImageSets/val.txt"

        title = f"{name}-{epoch}-deterministic-cls"
        if not (Path(save_dir)/f"{title}.pkl").exists():
            cmd0 = "python3 analyze_results.py draw_classification_calibration_plot "
            kwargs0 = dict(data_dir=str(data_dir),
                        anno_dir=str(anno_dir),
                        label_split_file=str(label_split_file),
                        save_path=str(Path(save_dir)/f"{title}.png"),
                        plot_data_path=str(Path(save_dir)/f"{title}.pkl"),
                        n_bins=10,
                        title=title, threshold=threshold)
            for k, v in kwargs0.items():
                cmd0 += f"--{k}={v} "
            cmds = [cmd0]
            return cmds
        else:
            return []

    names = ["pointpillar", "second", "pointrcnn"]
    epochs = ["7728", "7862", "7870"]
    cmd_list = []
    for name, epoch in zip(names, epochs):
        cmd_list += _assemble_cmds(name, epoch)
    run_cmd_list_single(cmd_list)
    # get all pkls
    data = {}
    for pkl_path in glob.glob(str(Path(save_dir)/"*-deterministic-cls.pkl")):
        pkl = read_pkl(pkl_path)
        name, _, mode, dim_name = Path(pkl_path).name[:-4].split("-")
        data.update({f"{name}-{mode}-{dim_name}":
            dict(x_vals=[float(itm) for itm in pkl["x_vals"]],
                 y_vals=[float(itm) if not np.isnan(itm) \
                     else 0.0 for itm in pkl["y_vals"]],
                 ece=pkl["ece"])})
    # plot
    modify_name={
        "pointpillar": "PP",
        "second": "SC",
        "pointrcnn": "PR"
    }
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times"],
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    mode = "deterministic"
    for name in ["pointpillar", "second", "pointrcnn"]:
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(1,1,1)
        for dim_name in ["cls"]:
            data_ = data[f"{name}-{mode}-{dim_name}"]
            ax.plot([0.0]+data_["x_vals"]+[1.0], [0.0]+data_["y_vals"]+[1.0], label=dim_name)
            ax.legend()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("Predicted confidence")
        plt.ylabel("Empirical confidence")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title(f"{modify_name[name]}-{mode}")
        plt.savefig(str(Path(save_dir)/f"{name}-{mode}.pdf"), dpi=100)

def plot_deterministic_ece_tables(result_data_dir):
    data = {}
    for pkl_path in glob.glob(str(Path(result_data_dir)/"*-deterministic-cls.pkl")):
        pkl = read_pkl(pkl_path)
        name, _, mode, dim_name = Path(pkl_path).name[:-4].split("-")
        data.update({f"{name}-{mode}-{dim_name}":
            dict(x_vals=[float(itm) for itm in pkl["x_vals"]],
                 y_vals=[float(itm) if not np.isnan(itm) \
                     else 0.0 for itm in pkl["y_vals"]],
                 ece=pkl["ece"])})

    modify_name={
        "pointpillar": "PP",
        "second": "SC",
        "pointrcnn": "PR"
    }
    dim_names = ["cls"]
    df_dict = {}
    mode = "deterministic"
    for name in ["pointpillar", "second", "pointrcnn"]:
        df_dict.update({
            f"{modify_name[name]}-{mode}": pd.Series([
            data[f"{name}-{mode}-{dim_name}"]["ece"]
            for dim_name in dim_names], index=dim_names)})
    df = pd.DataFrame(df_dict)
    print(df)
    writer = pd.ExcelWriter(str(Path(result_data_dir)/"ECE_table_deterministic.xlsx"), engine='xlsxwriter')
    df.to_excel(writer, sheet_name="expected calibration error")
    writer.save()

if __name__ == "__main__":
    data_dir = "/usr/app/openpcdet/data/kitti/training"
    validate_data_dir = "/usr/app/validate_data/"

    experiment_data_dir = str(Path(validate_data_dir)/"experiment_data")
    save_dir = str(Path(validate_data_dir)/"result_data"/"calibration_plots")
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    plot_deterministic_calibration_plots(save_dir, data_dir, experiment_data_dir)
    plot_deterministic_ece_tables(save_dir)
    plot_calibration_plots(save_dir, data_dir, experiment_data_dir)
    print_ece_table(result_data_dir=save_dir)
