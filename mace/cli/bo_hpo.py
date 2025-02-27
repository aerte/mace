# Bayesian Optimization of Hyperparameters

import argparse
import ast
import glob
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import torch.distributed
import torch.nn.functional
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, tools
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from mace.tools.model_script_utils import configure_model
from mace.tools.multihead_tools import (
    HeadConfig,
    assemble_mp_data,
    dict_head_to_dataclass,
    prepare_default_head,
)
from mace.tools.scripts_utils import (
    LRScheduler,
    check_path_ase_read,
    convert_to_json_format,
    dict_to_array,
    extract_config_mace_model,
    get_atomic_energies,
    get_avg_num_neighbors,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    get_loss_fn,
    get_optimizer,
    get_params_options,
    get_swa,
    print_git_commit,
    remove_pt_head,
    setup_wandb,
)
from mace.tools.slurm_distributed import DistributedEnvironment
from mace.tools.tables_utils import create_error_table
from mace.tools.utils import AtomicNumberTable

import wandb

import ax


def single_param_train_eval(r_max, args, input_dict, initial=False, log_wandb=False, iteration=0):

    #########################################################
    # Loading the input dictionary
    train_loader = input_dict['train_loader']
    valid_loaders = input_dict['valid_loaders']
    z_table = input_dict['z_table']
    heads = input_dict['heads']
    atomic_energies = input_dict['atomic_energies']
    model_foundation = input_dict['model_foundation']
    dipole_only = input_dict['dipole_only']
    tag = input_dict['tag']
    rank = input_dict['rank']
    train_sampler = input_dict['train_sampler']
    loss_fn = input_dict['loss_fn']
    device = input_dict['device']
    train_set = input_dict['train_set']
    local_rank = input_dict['local_rank']
    #########################################################
    # Update the r_max
    args.r_max = r_max
    #########################################################
    
    model, output_args = configure_model(args, train_loader, atomic_energies, model_foundation, heads, z_table)
    model.to(device)

    logging.debug(model)
    if initial:
        logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
        logging.info("")
        logging.info("===========OPTIMIZER INFORMATION===========")
        logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
        logging.info(f"Batch size: {args.batch_size}")
        if args.ema:
            logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
        logging.info(
            f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
        )
        logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
        logging.info(loss_fn)

    # Cueq
    if args.enable_cueq:
        if initial:
            logging.info("Converting model to CUEQ for accelerated training")
        assert model.__class__.__name__ in ["MACE", "ScaleShiftMACE"]
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)

    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    # If we run BO, we don't want to log to wandb
    #if args.wandb:
    #    setup_wandb(args)
    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
#        log_wandb=args.wandb,     # Let's just log to wandb if BO is false
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=train_sampler,
        rank=rank,
    )

    wandb_log_dict = {}
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = tools.evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        if rank == 0:
            if log_wandb:
                wandb_log_dict[valid_loader_name] = {
                    "iteration": iteration,
                    "valid_loss": valid_loss_head,
                    "valid_rmse_e_per_atom": eval_metrics[
                        "rmse_e_per_atom"
                    ],
                    "valid_rmse_f": eval_metrics["rmse_f"],
                }
    valid_loss = (
        valid_loss_head  # consider only the last head for the checkpoint
    )

    wandb.log(wandb_log_dict)

    # Optimize only over the loss for now
    #valid_loss_head = 0
    #for valid_loader_name, valid_loader in valid_loaders.items():
    #    valid_loss_head, eval_metrics = tools.evaluate(
    #        model=model,
    #        loss_fn=loss_fn,
    #        data_loader=valid_loader,
    #        output_args=output_args,
    #        device=device,
    #    )
    # valid_loss_head += valid_loss_head

    # Average over the valid loaders (just one if there is only one head)
    #valid_loss_head /= len(valid_loaders)
    #metrics, aux = tools.evaluate(model, loss_fn, valid_loaders, output_args, device)

    return {'loss': valid_loss}

