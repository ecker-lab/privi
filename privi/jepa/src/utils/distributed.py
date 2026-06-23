# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()

import socket

def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def broadcast_string(rank, string_to_broadcast=None, src=0, MAX_STR_LEN=500, device=None):

    if rank == src:
        # Encode and truncate string if needed
        encoded = string_to_broadcast.encode('utf-8')[:MAX_STR_LEN]
        length = len(encoded)
        send_tensor = torch.zeros(MAX_STR_LEN, dtype=torch.uint8, device=device)
        send_tensor[:length] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    else:
        send_tensor = torch.empty(MAX_STR_LEN, dtype=torch.uint8, device=device)
        length_tensor = torch.zeros(1, dtype=torch.long, device=device)

    # Broadcast length first
    dist.broadcast(length_tensor, src)

    # Broadcast fixed-size buffer
    dist.broadcast(send_tensor, src)

    # Truncate to actual content
    actual_length = length_tensor.item()
    return send_tensor[:actual_length].cpu().numpy().tobytes().decode('utf-8')

def init_distributed(port=37124, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        logger.info('Distributed training already initialized')
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            logger.info(f"Trying to init distributed training with SLURM vars {os.environ['SLURM_NTASKS']}, {os.environ['SLURM_PROCID']}")
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            #return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank
        )
        return world_size, rank
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'Rank: {rank}. Distributed training with fixed port not available {e}')

    try:
        # This is needed to run multiple single-gpu jobs on the same node
        os.environ['MASTER_PORT'] = str(get_free_port())
        logger.info(f'Rank: {rank}. Trying to init single-gpu fake distributed training with free port {os.environ["MASTER_PORT"]}')
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        logger.info(f'Rank: {rank}. Distributed training really not available {e}')

    return world_size, rank


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
