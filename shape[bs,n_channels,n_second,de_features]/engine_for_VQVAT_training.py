import math
import sys
from typing import Iterable
import torch
from torch import nn
from contextlib import nullcontext
import utils
import numpy as np


def train_one_epoch(model: torch.nn.Module,
                    vqdep: torch.nn.Module,
                    data_loader_list: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    ch_names_list=None,
                    args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',
                            utils.SmoothedValue(window_size=1,
                                                fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr',
                            utils.SmoothedValue(window_size=1,
                                                fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list,
                                     ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader,
                                                               print_freq * args.gradient_accumulation_steps,
                                                               header)):

            it = start_steps + step + step_loader
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = batch[0].float().to(device,
                                          non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_vectors = vqdep.get_codebook_quantize(samples,
                                                                input_chans)
                labels = batch[1].to(device,
                                     non_blocking=True)
            my_context = model.no_sync if args.distributed and (
                    step + 1) % args.gradient_accumulation_steps != 0 else nullcontext

            with my_context():
                with torch.cuda.amp.autocast():
                    outputs = model(input_vectors,
                                    input_chans)
                    loss = loss_fn(outputs,
                                   labels)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}",
                      force=True)
                sys.exit(1)

            is_second_order = hasattr(optimizer,
                                      'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss,
                                    optimizer,
                                    clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=is_second_order,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            Train_Acc = (outputs.max(-1)[1] == labels).float().mean().item()
            metric_logger.update(train_acc=Train_Acc)
            if log_writer is not None:
                log_writer.update(train_acc=Train_Acc,
                                  head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr,
                             group["lr"])
                max_lr = max(max_lr,
                             group["lr"])
            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value,
                                  head="loss")
                log_writer.update(loss_scale=loss_scale_value,
                                  head="opt")
                log_writer.update(lr=max_lr,
                                  head="opt")
                log_writer.update(min_lr=min_lr,
                                  head="opt")
                log_writer.update(weight_decay=weight_decay_value,
                                  head="opt")
                log_writer.update(grad_norm=grad_norm,
                                  head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:",
          metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader_list,
             model,
             vqdep,
             device,
             ch_names_list=None,
             metrics=None, ):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    pred = []
    true = []
    for data_loader, ch_names in zip(data_loader_list,
                                     ch_names_list):
        input_chans = utils.get_input_chans(ch_names)
        for step, batch in enumerate(metric_logger.log_every(data_loader,
                                                             10,
                                                             header)):
            samples = batch[0].float().to(device,
                                          non_blocking=True)
            labels = batch[1].to(device,
                                 non_blocking=True)
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_vectors = vqdep.get_codebook_quantize(samples,
                                                                input_chans)
                    output = model(input_vectors,
                                   input_chans)
                    loss = criterion(output,
                                     labels)

            results = utils.get_metrics(output,
                                        labels,
                                        metrics)
            pred.append(output)
            true.append(labels)

            batch_size = samples.shape[0]
            metric_logger.update(loss=loss.item())
            for key, value in results.items():
                metric_logger.meters[key].update(value,
                                                 n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    pred = torch.cat(pred,
                     dim=0)
    true = torch.cat(true,
                     dim=0)
    ret = utils.get_metrics(pred,
                            true,
                            metrics)
    ret['loss'] = metric_logger.loss.global_avg
    return ret
