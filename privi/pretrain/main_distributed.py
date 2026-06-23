# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Modified by Felix Benjamin Mueller, 2025

import argparse
import os
from pathlib import Path
import pprint
import yaml

import submitit

from privi.pretrain.scaffold import main as app_main
from privi.jepa.src.utils.logging import get_logger
from privi.utils.config import apply_overrides

logger = get_logger(force=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='output/submitit/backbone/')
parser.add_argument(
    '--exclude', type=str,
    help='nodes to exclude from training',
    default=None)
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str, default="grete",
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--time', type=int, default=2880,
    help='time in minutes to run job')
parser.add_argument(
        "-o",
        "--override",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        help="Override a config entry, e.g. -o data.path.train /new/train",
    )


class Trainer:

    def __init__(self, args_pretrain, load_model=None):
        self.app = args_pretrain['app']
        self.args_pretrain = args_pretrain
        self.load_model = load_model

    def __call__(self):
        app = self.app
        params = self.args_pretrain
        load_model = self.load_model

        logger.info('loaded pretrain params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

        # Launch app with loaded config
        resume_preempt = False if load_model is None else load_model
        app_main(app, args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_pretrain, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_app_with_parsed_args(
    args_for_pretrain,
    submitit_folder,
    partition,
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    exclude_nodes=None
):
    executor = submitit.AutoExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_mem_per_gpu='120G',
        slurm_constraint="ssd",
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=16,
        #slurm_gpus=f"A100:{tasks_per_node}",
        gpus_per_node=tasks_per_node,
        #slurm_gpus_per_task=1,
        #slurm_additional_parameters={"gres": "A100:4"},
        #slurm_gpus_per_node="A100:{tasks_per_node}",
        stderr_to_stdout=True,
        )

    if args.exclude is not None:
        executor.update_parameters(slurm_exclude=args.exclude)

    jobs, trainers = [], []
    with executor.batch():
        for ap in args_for_pretrain:
            fb_trainer = Trainer(ap)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


def launch():

    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
    # ---------------------------------------------------------------------- #


    overrides: list[tuple[str, str]] = []
    if args.override:
        overrides = [(k, v) for k, v in args.override]
    print("Overrides", overrides)

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, 'r') as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)
            _params["fname"] = f
            nodes = int(_params.get('nodes'))
            tasks_per_node = int(_params.get('tasks_per_node'))
            apply_overrides(_params, overrides)
            configs += [_params]
    logger.info(f'Loaded {len(configs)} config files')
    logger.info(f'Running all jobs with {nodes=} / {tasks_per_node=}')
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_app_with_parsed_args(
        args_for_pretrain=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
