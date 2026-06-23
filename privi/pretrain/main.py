# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Modified by Felix Benjamin Mueller, 2025

import argparse

import multiprocessing as mp

import pprint
import yaml

from privi.pretrain.scaffold import main as app_main
from privi.jepa.src.utils.distributed import init_distributed
from privi.utils.config import apply_overrides

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    help='which devices to use on local machine')
parser.add_argument(
        "-o",
        "--override",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        help="Override a config entry, e.g. -o data.path.train /new/train",
    )


def process_main(rank, fname, world_size, devices, overrides):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    from privi.jepa.src.utils.logging import get_logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        params["fname"] = fname
        logger.info('loaded params...')

    

    print("Overrodes", overrides)

    # then:
    apply_overrides(params, overrides)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    print("OVERRIDES START", args.override)

    overrides: list[tuple[str, str]] = []
    if args.override:
        overrides = [(k, v) for k, v in args.override]

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, overrides)
        ).start()
