import os
import time
import json
import datetime
import argparse
from pathlib import Path
import numpy as np
import torch
from timm import create_model
from torch.backends import cudnn
import utils
from engine_for_VQVAT_training import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import modeling_VAT_classifier
import modeling_vqdep


def get_args():
    parser = argparse.ArgumentParser(
            'VQ-VAT Train and Evaluate',
            add_help=False
    )
    # Classifier model settings
    parser.add_argument(
            '--model',
            default='VAT_classifier',
            type=str,
            metavar='MODEL',
            help='Classifier model'
    )
    # tokenizer settings
    parser.add_argument(
            "--tokenizer_weight",
            type=str
    )
    parser.add_argument(
            "--tokenizer_model",
            type=str,
            default="vqdep_vocab_1k_dim_32"
    )
    parser.add_argument(
            '--codebook_size',
            default=1024,
            type=int,
            help='number of codebook'
    )
    parser.add_argument(
            '--codebook_dim',
            default=32,
            type=int,
            help='dimension of codebook'
    )

    parser.add_argument(
            '--batch_size',
            default=32,
            type=int
    )
    parser.add_argument(
            '--epochs',
            default=101,
            type=int
    )
    parser.add_argument(
            '--save_ckpt_freq',
            default=20,
            type=int
    )

    # Optimizer parameters
    parser.add_argument(
            '--opt',
            default='adamw',
            type=str,
            metavar='OPTIMIZER',
            help='Optimizer (default: "adamw"'
    )
    parser.add_argument(
            '--opt_eps',
            default=1e-8,
            type=float,
            metavar='EPSILON',
            help='Optimizer Epsilon (default: 1e-8)'
    )
    parser.add_argument(
            '--opt_betas',
            default=None,
            type=float,
            nargs='+',
            metavar='BETA',
            help='Optimizer Betas (default: None, use opt default)'
    )
    parser.add_argument(
            '--clip_grad',
            type=float,
            default=None,
            metavar='NORM',
            help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            metavar='M',
            help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.05,
            help='weight decay (default: 0.05)'
    )
    parser.add_argument(
            '--weight_decay_end',
            type=float,
            default=None,
            help="""Final value of the weight decay. We use a cosine schedule for WD. 
            (Set the same value with args.weight_decay to keep weight decay no change)"""
    )

    parser.add_argument(
            '--lr',
            type=float,
            default=5e-4,
            metavar='LR',
            help='learning rate (default: 5e-4)'
    )
    parser.add_argument(
            '--warmup_lr',
            type=float,
            default=1e-6,
            metavar='LR',
            help='warmup learning rate (default: 1e-6)'
    )
    parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-5,
            metavar='LR',
            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)'
    )

    parser.add_argument(
            '--warmup_epochs',
            type=int,
            default=5,
            metavar='N',
            help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
            '--warmup_steps',
            type=int,
            default=-1,
            metavar='N',
            help='epochs to warmup LR, if scheduler supports'
    )

    parser.add_argument(
            '--output_dir',
            default='',
            help='path where to save, empty for no saving'
    )
    parser.add_argument(
            '--log_dir',
            default=None,
            help='path where to tensorboard log'
    )
    parser.add_argument(
            '--device',
            default='cuda',
            help='device to use for training / testing'
    )
    parser.add_argument(
            '--seed',
            default=0,
            type=int
    )
    parser.add_argument(
            '--resume',
            default='',
            help='resume from checkpoint'
    )
    parser.add_argument(
            '--auto_resume',
            action='store_true'
    )
    parser.add_argument(
            '--no_auto_resume',
            action='store_false',
            dest='auto_resume'
    )
    parser.set_defaults(
            auto_resume=True
    )

    parser.add_argument(
            '--save_ckpt',
            action='store_true'
    )
    parser.add_argument(
            '--no_save_ckpt',
            action='store_false',
            dest='save_ckpt'
    )
    parser.set_defaults(
            save_ckpt=True
    )

    parser.add_argument(
            '--dist_eval',
            action='store_true',
            default=True,
            help='Enabling distributed evaluation'
    )
    parser.add_argument(
            '--disable_eval',
            action='store_true',
            default=False
    )
    parser.add_argument(
            '--eval',
            action='store_true',
            default=False,
            help="Perform evaluation only"
    )

    parser.add_argument(
            '--start_epoch',
            default=0,
            type=int,
            metavar='N',
            help='start epoch'
    )
    parser.add_argument(
            '--num_workers',
            default=10,
            type=int
    )
    parser.add_argument(
            '--pin_mem',
            action='store_true',
            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
            '--no_pin_mem',
            action='store_false',
            dest='pin_mem',
            help=''
    )
    parser.set_defaults(
            pin_mem=True
    )

    # distributed training parameters
    parser.add_argument(
            '--world_size',
            default=1,
            type=int,
            help='number of distributed processes'
    )
    parser.add_argument(
            '--local_rank',
            default=-1,
            type=int
    )
    parser.add_argument(
            '--dist_on_itp',
            action='store_true'
    )
    parser.add_argument(
            '--dist_url',
            default='env://',
            help='url used to set up distributed training'
    )

    parser.add_argument(
            '--gradient_accumulation_steps',
            default=1,
            type=int
    )

    return parser.parse_args()


def get_model(args):
    model = create_model(args.model,
                         pretrained=False,
                         num_classes=args.num_classes,
                         feature_dim=args.codebook_dim, )
    return model


def get_tokenizer(args):
    print(f"Creating tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size,
            code_dim=args.codebook_dim, ).eval()
    return model


def get_dataset():
    dataset_train = [
            [Path("./PI/Dataset/SEED_PI/window5_step5")],
            [Path("./PI/Dataset/FACED_PI/window5_step5")],
    ]
    valance_train = [
            ['positive'],
            ['positive']
    ]
    dataset_train_list, train_ch_names_list, num_class = utils.build_dataset(dataset_train,
                                                                             valance_train,
                                                                             stage="train")
    print("Train num classes:",
          num_class)

    dataset_val = [
            [Path("./PI/Dataset/SEED_PI/window5_step5")],
            [Path("./PI/Dataset/FACED_PI/window5_step5")],
    ]
    valance_val = [
            ['positive'],
            ['positive']
    ]
    dataset_val_list, val_ch_names_list, num_class = utils.build_dataset(dataset_val,
                                                                         valance_val,
                                                                         stage="eval")
    print("Eval num classes:",
          num_class)

    metrics = ["accuracy", "balanced_accuracy"]

    return dataset_train_list, train_ch_names_list, dataset_val_list, val_ch_names_list, num_class, metrics


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train_list, train_ch_names_list, dataset_val_list, val_ch_names_list, num_class, metrics = get_dataset()
    if args.disable_eval:
        dataset_val_list = None

    args.num_classes = num_class

    print(args)

    model = get_model(args)
    vqdep = get_tokenizer(args).to(device)

    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(dataset,
                                                                num_replicas=num_tasks,
                                                                rank=sampler_rank,
                                                                shuffle=True)
            sampler_train_list.append(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))

        sampler_eval_list = []
        if args.dist_eval:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(dataset,
                                                                  num_replicas=num_tasks,
                                                                  rank=global_rank,
                                                                  shuffle=False)
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir,
                    exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list,
                                sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(dataset,
                                                        sampler=sampler,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        pin_memory=args.pin_mem,
                                                        drop_last=True, )
        data_loader_train_list.append(data_loader_train)

    if dataset_val_list is not None:
        data_loader_val_list = []
        for dataset, sampler in zip(dataset_val_list,
                                    sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(dataset,
                                                          sampler=sampler,
                                                          batch_size=int(1.5 * args.batch_size),
                                                          num_workers=args.num_workers,
                                                          pin_memory=args.pin_mem,
                                                          drop_last=False, )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of VAT's params:",
          n_parameters)

    print("Tokenizer = %s" % str(vqdep))
    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args,
                                 model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(args.lr,
                                                args.min_lr,
                                                args.epochs,
                                                num_training_steps_per_epoch,
                                                warmup_epochs=args.warmup_epochs,
                                                warmup_steps=args.warmup_steps, )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(args=args,
                          model=model,
                          model_without_ddp=model_without_ddp,
                          optimizer=optimizer,
                          loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val_list,
                              model,
                              vqdep,
                              device,
                              ch_names_list=val_ch_names_list,
                              metrics=metrics)
        print(f"Accuracy: {test_stats['accuracy']}, balanced accuracy: {test_stats['balanced_accuracy']}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch,
                       args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(model,
                                      vqdep,
                                      data_loader_train_list,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      args.clip_grad,
                                      log_writer=log_writer,
                                      start_steps=epoch * num_training_steps_per_epoch,
                                      lr_schedule_values=lr_schedule_values,
                                      wd_schedule_values=wd_schedule_values,
                                      ch_names_list=train_ch_names_list,
                                      args=args)
        if args.output_dir:
            utils.save_model(args=args,
                             model=model,
                             model_without_ddp=model_without_ddp,
                             optimizer=optimizer,
                             loss_scaler=loss_scaler,
                             epoch=epoch,
                             save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val_list is not None:
            val_stats = evaluate(data_loader_val_list,
                                 model,
                                 vqdep,
                                 device,
                                 ch_names_list=val_ch_names_list,
                                 metrics=metrics)

            print(f"Accuracy of the network on the {sum([len(dataset) for dataset in dataset_val_list])} "
                  f"val EEG: {val_stats['accuracy']:.4f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(args=args,
                                     model=model,
                                     model_without_ddp=model_without_ddp,
                                     optimizer=optimizer,
                                     loss_scaler=loss_scaler,
                                     epoch="best")
            print(f'Max accuracy val: {max_accuracy:.4f}%')

            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value,
                                          head="val",
                                          step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value,
                                          head="val",
                                          step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value,
                                          head="val",
                                          step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir,
                                   "log.txt"),
                      mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True,
                                    exist_ok=True)
    main(opts)
