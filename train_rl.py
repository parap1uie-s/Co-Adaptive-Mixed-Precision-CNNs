import math
import time
import os
import traceback
import logging
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from distiller.data_loggers import *
import distiller.quantization as quantization

from utils import *
import copy
# from distiller.models import ALL_MODEL_NAMES, create_model
from models import ALL_MODEL_NAMES, create_model
from knowledge_distillation import KnowledgeDistillationPolicy
import parser
import os

# hack
distiller.KnowledgeDistillationPolicy = KnowledgeDistillationPolicy
# hack distiller
from data_loaders import load_data
apputils.load_data = load_data

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'

class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()
    def forward(self, x, target):
        logp = F.log_softmax(x, dim=1)
        target = target.view(-1,1)
        output = - logp.gather(1, target)
        return output

def train(train_loader, model, criterion, total_criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    global gate_saved_actions
    global gate_rewards

    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit
    # So exiterrors is analogous to classerr for the non-Early Exit case
    if args.earlyexit_lossweights:
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        if not hasattr(args, 'kd_policy') or args.kd_policy is None or (not args.kd_policy.active):
            output = model(inputs)
        else:
            output, tfm, sfm = args.kd_policy.forward(inputs) # teacher/student feature map

        if isinstance(output, list):
            output = output[0]

        pred_loss = criterion(output.data, target)
        # Measure accuracy
        classerr.add(output.data, target)
        acc_stats.append([classerr.value(1), classerr.value(5)])

        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(pred_loss.mean().item())

        total_loss = total_criterion(output, target)
        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, total_loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            total_loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(total_loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(total_loss.item())

        # if args.kd_policy and args.kd_policy.active:
        #     pred_loss = args.kd_policy.get_pred_loss(pred_loss)
        #     for k in range(len(sfm)):
        #         total_loss += 0.3 * F.mse_loss(sfm[k], tfm[k])

        #############
        ###  RL start 
        # re-weight gate rewards
        normalized_alpha = args.alpha / len(gate_saved_actions)
        # intermediate rewards for each gate
        for (log_prob, action) in gate_saved_actions:
            # m, act = item
            gate_rewards.append((1 - action.float()).data * normalized_alpha)
        # pdb.set_trace()
        # collect cumulative future rewards
        R = - pred_loss.data
        cum_rewards = []
        for r in gate_rewards[::-1]:
            R = r + args.gamma * R
            cum_rewards.insert(0, R)

        # apply REINFORCE to each gate
        # Pytorch 2.0 version. `reinforce` function got removed in Pytorch 3.0
        rl_loss = []
        for (log_prob, action), R in zip(gate_saved_actions, cum_rewards):
             # action.reinforce(args.rl_weight * R)
            rl_loss.append( torch.mean(-log_prob * args.rl_weight * R) ) 

        total_loss += torch.stack(rl_loss).sum()

        ###  RL end
        #############

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                errs['Top1'] = classerr.value(1)
                errs['Top5'] = classerr.value(5)
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                    errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()

        # clear saved actions and rewards
        del gate_saved_actions[:]
        del gate_rewards[:]
    return acc_stats

def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if args.earlyexit_thresholds:
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            if args.kd_policy:
                output, _ = model(inputs)
            else:
                output = model(inputs)

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                if not args.earlyexit_thresholds:
                    stats = ('',
                            OrderedDict([('Loss', losses['objective_loss'].mean),
                                         ('Top1', classerr.value(1)),
                                         ('Top5', classerr.value(5))]))
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                        # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                        # accordingly and we will not print for an exit error when that exit is never taken.
                        if args.exit_taken[exitnum]:
                            t1 = 'Top1_exit' + str(exitnum)
                            t5 = 'Top5_exit' + str(exitnum)
                            stats_dict[t1] = args.exiterrors[exitnum].value(1)
                            stats_dict[t5] = args.exiterrors[exitnum].value(5)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]

def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir, msglogger)
    return top1, top5, lossses

def main():
    script_dir = os.path.dirname(__file__)
    global msglogger
    global gate_saved_actions
    global gate_rewards

    # Parse arguments
    args = parser.get_parser().parse_args()
    args.device = 'cuda'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    start_epoch = 0
    perf_scores_history = []

    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    # args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    if args.dataset == 'cifar10':
        args.num_classes = 10 
    elif args.dataset == 'cifar100':
        args.num_classes = 100 
    else:
        args.num_classes = 1000

    # Create the model
    basemodel = create_model(False, args.dataset, args.arch[:args.arch.rindex("_")], parallel=not args.load_serialized, device_ids=[0], gate_channel=args.gate_channel, pre_gate=args.pre_gate)
    RLmodel = create_model(args.pretrained, args.dataset, args.arch, parallel=not args.load_serialized, device_ids=args.gpus, gate_channel=args.gate_channel, pre_gate=args.pre_gate)

    gate_saved_actions = RLmodel.module.saved_actions
    gate_rewards = RLmodel.module.rewards

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # resume sp trained base model
    if args.resume:
        basemodel, _, _ = apputils.load_checkpoint(basemodel, chkpt_file=args.resume)
        basemodel.to(args.device)

    # Define loss function (criterion) and optimizer
    criterion = BatchCrossEntropy().to(args.device)
    total_criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, RLmodel.parameters()), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    activations_collectors = create_activation_stats_collectors(RLmodel, *args.activation_stats)

    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)

    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(RLmodel, optimizer, args.compress, compression_scheduler)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        RLmodel.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(RLmodel)

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=[0], deeply_supervised=True)
        if args.kd_resume:
            teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)

        RLmodel.module.fc1 = teacher.module.fc1
        RLmodel.module.fc2 = teacher.module.fc2
        
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(RLmodel, teacher, args.kd_temp, dlw, rl=True)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

    RLmodel.load_state_dict(copy.deepcopy(basemodel.state_dict()), strict=False)

    for i in range(len(RLmodel.module.layer1)):
        RLmodel.module.layer1[i].fullbit_block.conv1.float_weight = RLmodel.module.layer1[i].basic_block.conv1.float_weight
        RLmodel.module.layer1[i].fullbit_block.conv2.float_weight = RLmodel.module.layer1[i].basic_block.conv2.float_weight
        RLmodel.module.layer2[i].fullbit_block.conv1.float_weight = RLmodel.module.layer2[i].basic_block.conv1.float_weight
        RLmodel.module.layer2[i].fullbit_block.conv2.float_weight = RLmodel.module.layer2[i].basic_block.conv2.float_weight
        RLmodel.module.layer3[i].fullbit_block.conv1.float_weight = RLmodel.module.layer3[i].basic_block.conv1.float_weight
        RLmodel.module.layer3[i].fullbit_block.conv2.float_weight = RLmodel.module.layer3[i].basic_block.conv2.float_weight

    del(basemodel)

    #############
    ### train ###
    #############
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, RLmodel, criterion, total_criterion, optimizer, epoch, compression_scheduler,
                  loggers=[pylogger], args=args)
            # distiller.log_weights_sparsity(model, epoch, loggers=[pylogger])
            # distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                # collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(RLmodel, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, RLmodel, total_criterion, [pylogger], args, epoch)
            # distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
            #                                     collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir, msglogger)

        # clear saved actions and rewards
        del gate_saved_actions[:]
        del gate_rewards[:]

        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[pylogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, RLmodel, top1, top5, epoch, args.num_best_scores, msglogger)
        is_best = epoch == perf_scores_history[0].epoch
        apputils.save_checkpoint(epoch, args.arch, RLmodel, optimizer, compression_scheduler,
                                 perf_scores_history[0].top1, is_best, args.name, msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, RLmodel, total_criterion, [pylogger], activations_collectors, args=args)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Logger handle
    msglogger = None
    gate_saved_actions = None
    gate_rewards = None
    main()