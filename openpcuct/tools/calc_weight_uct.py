'''
This file is modified from tools/test.py.
NAME=pointrcnn
EPOCH=7870
CKPT_DIR=/usr/app/OpenPCDet
cd $ROOT_DIR/tools && python3 calc_weight_uct.py \
    --cfg cfgs/kitti_models/${NAME}.yaml \
    --ckpt ${CKPT_DIR}/${NAME}_${EPOCH}.pth \
    --batch_size 1
'''
import re, os, datetime
from pathlib import Path

from test import log_config_to_file, \
    build_dataloader, build_network, \
    parse_config, common_utils
from test import parse_config as test_parse_config

from pcuct.laplace_approx.fisher.fisher import \
    get_setup_fn, get_closure_fn, \
    compute_mean, compute_fisher, \
    save_weight_distribution

def parse_config():
    args, cfg = test_parse_config()
    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist = False
        total_gpus = 1
    else:
        err_msg = "Do not support args.launcher."
        assert False, err_msg
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.fix_random_seed:
        common_utils.set_random_seed(args.random_seed)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        err_msg = "Do not support args.eval_all."
        assert False, err_msg
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist, workers=args.workers, logger=logger, training=True
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    logger.info('Start compute mean & fisher.')
    mean_dict, fisher_dict = compute_weight_distribution(
        model, dataloader, args, logger, dist)
    path = Path(eval_output_dir)/"pdist.pkl"
    save_weight_distribution(
        str(path), mean_dict, fisher_dict)
    logger.info(f'Save mean & fisher into {str(path)}')

def compute_weight_distribution(
    model,
    dataloader,
    args,
    logger,
    dist):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)
    model.cuda()

    param_dict = {}
    closure_fn = get_closure_fn(
        name=str(Path(args.cfg_file).name).split('.')[0],
        mode=args.fisher_mode)
    setup_fn = get_setup_fn()
    mean_dict = compute_mean(model)
    fisher_dict = compute_fisher(
        model, dataloader, closure_fn, setup_fn, param_dict)
    
    return mean_dict, fisher_dict

if __name__ == "__main__":
    main()