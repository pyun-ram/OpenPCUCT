import os
import sys
import time
import glob
import fire
from det3.ops.ops import write_pkl
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from det3.ops import read_pkl, read_txt, read_img
from pcuct.utils import uct_calib_utils

def _parse_text_kernel(text, data_dict, idx_start, mode):
    '''
    Kernel function of _parse_text
    Args:
        text (List[str])
        data_dict (Dict)
        idx_start (int)
        mode (str): 'bev'/'3d'
    '''
    tmp = {}
    tmp['Car'] = text[idx_start].split(":")[-1].split(",")
    tmp['Pedestrian'] = text[idx_start+20].split(":")[-1].split(",")
    tmp['Cyclist'] = text[idx_start+40].split(":")[-1].split(",")
    for cls in ['Car', 'Pedestrian', 'Cyclist']:
        tmp_ = {}
        tmp_['easy'], tmp_['moderate'], tmp_['hard'] = tmp[cls]
        for level in ['easy', 'moderate', 'hard']:
            data_dict[f'{cls}_{mode}/{level}_R40'] = float(tmp_[level])

def _parse_text(text):
    '''
    Parse text into data_dict
    Args:
        text (List[str])
    Return:
        data_dict (Dict)
    '''
    idx_start = len(text)
    for i in range(idx_start):
        s = text[len(text)-i-1]
        if "Car AP@0.70, 0.70, 0.70:" in s:
            idx_start = len(text)-i-1
    data_dict = {}
    # 7 is the line number of bev mAP in <text>
    _parse_text_kernel(text, data_dict, idx_start+7, "bev")
    # 13 is the line number of 3d mAP in <text>
    _parse_text_kernel(text, data_dict, idx_start+8, "3d")
    # 17 is the line number of bev mAP (looser threshold) in <text>
    _parse_text_kernel(text, data_dict, idx_start+17, "bev_loose")
    # 18 is the line number of 3d mAP (looser threshold) in <text>
    _parse_text_kernel(text, data_dict, idx_start+18, "3d_loose")
    return data_dict

class ResultsAnalyzer:
    def draw_classification_calibration_plot(
        self,
        data_dir,
        anno_dir,
        label_split_file,
        save_path,
        plot_data_path=None,
        title="",
        n_bins=20,
        threshold=0.7):
        scores, preds, labels = uct_calib_utils.match_gt_and_det_classification(
            data_dir, anno_dir, label_split_file, threshold)
        uct_calib_utils.make_model_diagrams_classification(
            scores, preds, labels, n_bins, plot_data_path=plot_data_path)
        plt.title(title)
        plt.savefig(save_path, bbox_inches="tight", dpi=100)

    def draw_regression_calibration_plot(
        self,
        data_dir,
        anno_dir,
        uct_dir,
        label_split_file,
        save_path,
        plot_data_path=None,
        title="",
        n_bins=20,
        dim=0,
        threshold=0.7):
        covs, preds, labels = uct_calib_utils.match_gt_and_det_regression(
            data_dir, anno_dir, uct_dir, label_split_file, dim, threshold)
        uct_calib_utils.make_model_diagrams_regression(
            covs, preds, labels, n_bins, plot_data_path=plot_data_path)
        plt.title(title)
        plt.savefig(save_path, bbox_inches="tight")

    def draw_multivariate_regression_calibration_plot(
        self,
        data_dir,
        anno_dir,
        uct_dir,
        label_split_file,
        save_path,
        title="",
        n_bins=20,
        dims=0,
        threshold=0.7):
        covs, preds, labels = \
            uct_calib_utils.match_gt_and_det_multivariate_regression(
            data_dir, anno_dir, uct_dir, label_split_file, dims, threshold)
        uct_calib_utils.make_model_diagrams_multivariate_regression(
            covs, preds, labels, n_bins)
        plt.title(title)
        plt.savefig(save_path, bbox_inches="tight")

    def visualize_data(
        self,
        path,
        glob_cmd,
        title,
        save_path):
        '''
        Download image from local, and plot them in a single pdf file.
        '''
        def _get_figure_dir(path):
            t, path = path.split(":")
            if t == "local":
                return path
            else:
                raise NotImplementedError

        local_dir = _get_figure_dir(path)
        print(f"Run glob: {Path(local_dir)/glob_cmd}")
        image_paths = glob.glob(
            f"{Path(local_dir)/glob_cmd}", recursive=True)
        print(f"Found {len(image_paths)} images.")
        M = len(image_paths)
        num_col = min(3, M)
        num_row = int(M / 3) + 1
        plt.figure(figsize=(10 * num_row, 10 * num_col), dpi=100)
        for i, (image_path) in enumerate(image_paths):
            img = read_img(image_path)
            ax = plt.subplot(num_row, num_col, i+1)
            ax.imshow(img)
            ax.set_title(image_path.split("/")[-1])
        plt.suptitle(title)
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        print(f"Save the plot into {save_path} ")

    def plot_relationship(
        self,
        paths,
        tags,
        x_key,
        x_vals,
        y_keys,
        title,
        save_path,
        merge=False,
        x_logscale=False,
        save_plot_data=False,
        y_lim=(0,100)):
        N = len(paths)
        assert N == len(tags) == len(x_vals)
        M = len(y_keys) if not merge else 1
        data_list = [self._get_mAP_data(tag, path)
            for path, tag in zip(paths, tags)]
        print(f"The available keys include")
        for k in data_list[0].keys():
            print(k)

        # process x
        if x_logscale:
            x_ticks = [x for x in x_vals]
            x_vals = [np.log10(x) for x in x_vals]
        else:
            x_ticks = x_vals
        # process y
        y_vals_list = []
        for y_key in y_keys:
            y_vals_list.append([data_dict[y_key]
                for data_dict in data_list])
        if merge:
            # merge by averaging
            y_vals_merge = np.array(y_vals_list).mean(axis=0)
            y_vals_list = [y_vals_merge]
            print(f"Merge the keys: {', '.join(y_keys)}")

        num_col = min(3, M)
        num_row = int(M / 3) + 1
        plt.figure(figsize=(5 * num_row, 8 * num_col))
        for i, (y_vals, y_key) in enumerate(zip(y_vals_list, y_keys)):
            ax = plt.subplot(num_row, num_col, i+1)
            ax.plot(x_vals, y_vals, '.-')
            ax.set_title(title)
            ax.set_xlabel(x_key)
            ax.set_xticks(x_vals) 
            ax.set_xticklabels(x_ticks)
            ax.set_ylabel(y_key)
            ax.set_ylim(y_lim)

        plt.title(title)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Save the plot into {save_path} ")
        if save_plot_data:
            plot_data = dict(
                x_key=x_key,
                x_vals=x_vals,
                y_keys=y_keys,
                y_vals_list=y_vals_list,
                title=title)
            save_path_surf = ".".join(save_path.split(".")[:-1])
            write_pkl(plot_data, save_path_surf+".pkl")

    def compare_mAP_tables(
        self,
        paths,
        tags,
        classes=None,
        title="",
        use_looser_threshold=False,
        save_path=None):
        N = len(paths)
        fig = plt.figure(figsize=(8*N,2))
        xlsx_path = "/".join(
            save_path.split("/")[:-1]
            + [f"{title}.xlsx"])
        writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
        for i, (tag, path) in enumerate(zip(tags, paths)):
            data_dict = self._get_mAP_data(tag, path)
            ax = fig.add_subplot(1, N, i+1)
            df, ax = self._render_IoU_table(
                data_dict,
                classes,
                use_looser_threshold,
                ax=ax)
            df.to_excel(writer, sheet_name=data_dict["tag"][:30])
        fig.suptitle(title)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Save the table into {save_path}")
        writer.save()
        print(f"Save the table into {xlsx_path}")

    def get_mAP_table(
        self,
        path,
        tag,
        classes=None,
        use_looser_threshold=False,
        save_path=None):
        data_dict = self._get_mAP_data(tag, path)
        if save_path is None:
            ax=None
        else:
            fig = plt.figure(figsize=(8,2))
            ax = fig.add_subplot(111)
        df, ax = self._render_IoU_table(
            data_dict,
            classes,
            use_looser_threshold,
            ax=ax)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Save the table into {save_path}")
            xlsx_path=save_path[:-4]+".xlsx"
            writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
            df.to_excel(writer, 'mAP', startcol=0, startrow=1)
            _, n_cols = df.shape
            writer.sheets['mAP'].merge_range(0, 0, 0, n_cols, data_dict["tag"])
            writer.save()
            print(f"Save the table into {xlsx_path}")

    def _render_IoU_table(
        self,
        data_dict,
        classes,
        use_looser_threshold,
        ax=None):
        if classes is None:
            classes = ["Car", "Pedestrian", "Cyclist"]
        df_dict = {}
        for mode in ["BEV", "3D"]:
            for level in ["easy", "moderate", "hard"]:
                _modify_level = lambda l: l if l is not "moderate" else "mod."
                _modify_mode = lambda m: m.lower() if not use_looser_threshold \
                    else f"{m.lower()}_loose"
                df_dict.update({
                    f"{mode}-{_modify_level(level)}":pd.Series([
                        data_dict[f"{cls}_{_modify_mode(mode)}/{level}_R40"]
                        for cls in classes], index=classes)})
        df = pd.DataFrame(df_dict)
        if ax is None:
            print(df)
        else:
            ax.table(cellText = df.values,
                    rowLabels = df.index,
                    colLabels = df.columns,
                    loc = "center")
            ax.set_title(f"{data_dict['tag']}")
            ax.axis("off");
        return df, ax

    def _get_mAP_data(
        self,
        tag,
        path):
        '''
        Args:
            tag (str)
            path (str): 'local:<path>'
        Returns:
            data_dict (Dict)
        '''
        t, path = path.split(":")
        filename = path.split("/")[-1]
        if filename == "eval_uct_result.pkl":
            data_dict = read_pkl(path)
        elif filename in ["eval_uct_result.txt", "output.log"]:
            text = read_txt(path)
            try:
                data_dict = _parse_text(text)
            except IndexError:
                print(f"Cannot parse {tag}")
                sys.exit("Exit due to error")
        elif filename[:3] == "log" and filename[-3:] == "txt":
            text = read_txt(path)
            data_dict = _parse_text(text)
        else:
            raise NotImplementedError
        data_dict.update({"tag": tag})
        return data_dict

if __name__ == "__main__":
    fire.Fire(ResultsAnalyzer)