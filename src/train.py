import logging
from pathlib import Path
import argparse
import wandb
from .anchor_based.train import train as train_anchor_based
from .anchor_free.train import train as train_anchor_free
from .helpers import init_helper, data_helper
import torch


import numpy as np
TRAINER = {
    'anchor-based': train_anchor_based,
    'anchor-free': train_anchor_free
}

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        choices=('anchor-based', 'anchor-free'))
    parser.add_argument('--dataset', type=str, default='tvsum')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--nms-thresh', type=float, default=0.5)

    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    parser.add_argument('--neg-sample-ratio', type=float, default=2.0)
    parser.add_argument('--incomplete-sample-ratio', type=float, default=1.0)
    parser.add_argument('--pos-iou-thresh', type=float, default=0.6)
    parser.add_argument('--neg-iou-thresh', type=float, default=0.0)
    parser.add_argument('--incomplete-iou-thresh', type=float, default=0.3)
    parser.add_argument('--anchor-scales', type=int, nargs='+',
                        default=[4, 8, 16, 32])
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--cls-loss', type=str, default='focal',
                        choices=['focal', 'cross-entropy'])
    parser.add_argument('--reg-loss', type=str, default='soft-iou',
                        choices=['soft-iou', 'smooth-l1'])
    
    parser.add_argument('--wandb-key', '-wk', type = str, default=None)

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args


def get_trainer(model_type):
    assert model_type in TRAINER
    return TRAINER[model_type]


def main():
    args = get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)


    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(args.model)

    data_helper.dump_yaml(vars(args), model_dir / 'args.yml')
    
    
    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        results = {}
        stats = data_helper.AverageMeter('fscore')

        for split_idx, split in enumerate(splits):
            # logger.info(f'Start training on {split_path.stem}: split {split_idx}')
            ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)
            fscore = trainer( args, split, ckpt_path, args.wandb_key)
            stats.update(fscore=fscore)
            results[f'split{split_idx}'] = float(fscore)
            
            
        results['mean'] = float(stats.fscore)
        data_helper.dump_yaml(results, model_dir / f'{split_path.stem}.yml')
        

if __name__ == '__main__':
    main()
    