import os
from pathlib import Path

def run_cmd_list_single(cmd_list):
    for cmd in cmd_list:
        os.system(cmd)
    return

name2epoch = {
    "pointpillar": "7728",
    "second": "7862",
    "pointrcnn": "7870"
}
alpha_dict = {
    "standard": 0.003,
    "empirical": 0.01
}
dropout_dict = {
    "pointpillar": 0.05,
    "second": 0.2,
    "pointrcnn": 0.4
}

def run_deterministic_model(name):
    def _assemble_cmd(name, epoch):
        exp_prefix = f"{name}-deterministic-seed666"
        ckpt_dir = "/usr/app/openpcdet/"
        cmd = "python3 deploy_experiments.py generate_deterministic_predictions "
        kwargs = dict(
            exp_name=exp_prefix+"-gen_det_pred",
            extra_tag=exp_prefix+"-gen_det_pred",
            cfg=f"cfgs/kitti_models/{name}.yaml",
            ckpt=f"{ckpt_dir}/{name}_{epoch}.pth",
            fix_random_seed=True,
            random_seed=666,
            debug=False)
        for k, v in kwargs.items():
            cmd += f"--{k}={v} "
        return [cmd]
    cmds = _assemble_cmd(name, name2epoch[name])
    run_cmd_list_single(cmds)

def calculate_fisher(name, random_seed, modes):
    def _assemble_cmds(name, mode):
        epoch = name2epoch[name]
        exp_pref = f"{name}-{mode}-seed{random_seed}"
        ckpt_dir = "/usr/app/openpcdet"
        cmd0 = "python3 deploy_experiments.py calculate_fisher "
        kwargs0 = dict(
            exp_name=exp_pref+"-calculate_fisher",
            extra_tag=exp_pref+"-calculate_fisher",
            cfg=f"cfgs/kitti_models/{name}.yaml",
            ckpt=f"{ckpt_dir}/{name}_{epoch}.pth",
            fisher_mode=mode,
            fix_random_seed=True,
            random_seed=random_seed,
            debug=False)
        for k, v in kwargs0.items():
            cmd0 += f"--{k}={v} "
        return [cmd0]

    cmd_list = []
    for mode in modes:
        cmds = _assemble_cmds(name, mode)
        cmd_list += cmds
    run_cmd_list_single(cmd_list)

def bayesian_inference(name, random_seed, bnn_modes):
    def _assemble_cmds(name, mode, bnn_mode):
        epoch = name2epoch[name]
        alpha = alpha_dict[mode]
        dropout_rate = dropout_dict[name]
        exp_pref = f"{name}-{mode}-seed{random_seed}-{bnn_mode}"
        fisher_tag =  f"{name}-{mode}-seed{random_seed}-calculate_fisher"
        root_dir = "/usr/app/openpcdet"
        ckpt_dir = "/usr/app/openpcdet"
        data_dir = str(Path(root_dir)/"data"/"kitti"/"training")

        num_mc_samples = 10
        pdist_prior_cls = 1e-2
        pdist_prior_box = 1e-3
        kwargs1 = dict(
            exp_name=exp_pref+"-gen_pred_dist",
            extra_tag=exp_pref+"-gen_pred_dist",
            cfg=f"cfgs/kitti_models/{name}.yaml",
            ckpt=f"{ckpt_dir}/{name}_{epoch}.pth",
            weight_dist_path=f"{root_dir}/output/kitti_models/{name}/{fisher_tag}/eval/epoch_{epoch}/val/default/pdist.pkl",
            fisher_mode=mode,
            bayesian_inference_mode=bnn_mode,
            alpha=alpha,
            dropout_rate=dropout_rate,
            scaler_weight_prior=1e-4, # dummy parameter
            num_MC_samples=num_mc_samples,
            pdist_prior_cls=pdist_prior_cls,
            pdist_prior_box=pdist_prior_box,
            debug=False,
            fix_random_seed=True,
            random_seed=random_seed)

        cmd1 = "python3 deploy_experiments.py generate_predictive_distributions "
        for k, v in kwargs1.items():
            cmd1 += f"--{k}={v} "

        pred_dir = Path(root_dir)/"output/kitti_models"/name
        pred_dir = Path(pred_dir)/f"{exp_pref}-gen_pred_dist"
        pred_dir = Path(pred_dir)/"eval"/f"epoch_{epoch}"/f"val/default/final_result"
        cmd2 = "python3 deploy_experiments.py evaluate_predictive_distributions "
        kwargs2 = dict(
            exp_name=exp_pref+"-eval_pred_dist",
            data_dir=data_dir,
            pred_dir=pred_dir,
            debug=False)
        for k, v in kwargs2.items():
            cmd2 += f"--{k}={v} "
        return [cmd1, cmd2]

    modes_dict = {
        "mcdropout": ["standard"],
        "last-layer": ["standard", "empirical"],
        "last-module": ["standard", "empirical"],
        "full-net": ["standard", "empirical"]
    }

    cmd_list = []
    for bnn_mode in bnn_modes:
        for mode in modes_dict[bnn_mode]:
            cmds = _assemble_cmds(name, mode, bnn_mode)
            cmd_list += cmds
    run_cmd_list_single(cmd_list)

def conduct_experiments(random_seed, name, bnn_modes):
    calculate_fisher(name, random_seed,
        modes=["standard", "empirical"])
    bayesian_inference(name, random_seed,
        bnn_modes=bnn_modes)

if __name__ == "__main__":
    name = "pointpillar"
    bnn_modes=["mcdropout", "last-layer", "full-net"]
    run_deterministic_model(name)
    conduct_experiments(666, name, bnn_modes)

    name = "second"
    bnn_modes=["mcdropout", "last-layer", "full-net"]
    run_deterministic_model(name)
    conduct_experiments(666, name, bnn_modes)

    name = "pointrcnn"
    bnn_modes=["mcdropout", "last-layer", "last-module", "full-net"]
    run_deterministic_model(name)
    conduct_experiments(666, name, bnn_modes)
