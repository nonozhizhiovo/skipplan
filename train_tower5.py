import argparse
import os
import parser
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss

from data_utils.data_loader import dataset
from callbacks import AverageMeter, Logger, set_save_path, error_set_save_path
import time
import numpy as np

from collections import OrderedDict
import shutil
from utils import *

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import pickle

import collections
from datasets.CrossTask_args import parse_args
from datasets.CrossTask_dataloader import *

from focalloss import *


parser = argparse.ArgumentParser()

data_path = "datasets/CrossTask_assets"

parser.add_argument(
    "--data_path", type=str, default=data_path, help="default data path"
)

parser.add_argument(
    "--primary_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/tasks_primary.txt"),
    help="list of primary tasks",
)
parser.add_argument(
    "--related_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/tasks_related.txt"),
    help="list of related tasks",
)
parser.add_argument(
    "--annotation_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/annotations"),
    help="path to annotations",
)
parser.add_argument(
    "--video_csv_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/videos.csv"),
    help="path to video csv",
)
parser.add_argument(
    "--val_csv_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/videos_val.csv"),
    help="path to validation csv",
)
parser.add_argument(
    "--features_path",
    type=str,
    default=os.path.join(data_path, "crosstask_features"),
    help="path to features",
)
parser.add_argument(
    "--constraints_path",
    type=str,
    default=os.path.join(data_path, "crosstask_constraints"),
    help="path to constraints",
)
parser.add_argument(
    "--n_train", type=int, default=30, help="videos per task for training"
)

parser.add_argument(
    "--use_related",
    type=int,
    default=0,
    help="1 for using related tasks during training, 0 for using primary tasks only",
)
parser.add_argument(
    "--share",
    type=str,
    default="words",
    help="Level of sharing between tasks",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="crosstask",
    help="Used dataset name for logging",
)
parser.add_argument(
    "--dataloader-type",
    type=str,
    default="ddn",
    help="The type of dataset processing loader: either ddn or plate",
)
parser.add_argument(
    "--label-type",
    type=str,
    default="ddn",
    help="The type of dataset processing loader: either ddn or plate",
)

parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--max_traj_len', default=5, type=int, help='action number')
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--dataset_root', default='./crosstask/')
parser.add_argument('--frameduration', default=3, type=int)
parser.add_argument('--dataset_mode', default='multiple')
parser.add_argument('--frame_mode', default='part')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--validation_split', default=0.2, type=float)

parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--start_epoch', default=None, type=int)
parser.add_argument('--lr_steps', default=[100, 150, 200, 250, 300, 350, 400, 450, 500, 700, 900], type=float)

parser.add_argument('--clip_gradient', default=5, type=float)

parser.add_argument('--print_freq', '-p', default=100, type=int, help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int, help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--memory', default=False, type=str)
parser.add_argument('--auto_regressive', default=False)
parser.add_argument('--memory_size', default=128)
parser.add_argument('--selflearned_memory', default=True, type=bool)

parser.add_argument('--N', default=1, type=int,help='Number of layers in the temporal decoder')
parser.add_argument('--H', default=16, type=int,help='Number of heads in the temporal decoder')
parser.add_argument('--d_model', default=1024, type=int)
parser.add_argument('--decoder_dropout', default=0, type=float)
parser.add_argument('--feat_dropout', default=0, type=float)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=99999999, type=int)

parser.add_argument('--exist_datasplit', default=False, type=bool)
parser.add_argument('--contra_loss', default=False, type=bool)

parser.add_argument('--dim_feedforward', default=1024, type=int)

parser.add_argument('--mlp_mid', default=512, type=int)

parser.add_argument('--feat_mid', default=1024, type=int)

parser.add_argument('--query_length', default=6, type=int)
parser.add_argument('--memory_length', default=6, type=int)
parser.add_argument('--init_weight', default=True, type=bool)
parser.add_argument('--gamma', default=1.5, type=float)
parser.add_argument('--smallmid_ratio', default=3, type=int)

# options
args = parser.parse_args()

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.set_default_tensor_type('torch.FloatTensor')

best_loss = 1000000
best_acc = -np.inf
best_success_rate = -np.inf
best_miou = -np.inf

########################################
# Start Loading/Processing the dataset #
########################################

task_vids = get_vids(args.video_csv_path)
val_vids = get_vids(args.val_csv_path)
task_vids = {
    task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]]
    for task, vids in task_vids.items()
}
primary_info = read_task_info(args.primary_path)
test_tasks = set(primary_info["steps"].keys())
if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info["steps"], **related_info["steps"]}
    n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
else:
    task_steps = primary_info["steps"]
    n_steps = primary_info["n_steps"]
all_tasks = set(n_steps.keys())
task_vids = {task: vids for task,
             vids in task_vids.items() if task in all_tasks}
val_vids = {task: vids for task, vids in val_vids.items() if task in all_tasks}

with open(os.path.join(args.data_path, "crosstask_release/cls_step.json"), "r") as f:
    step_cls = json.load(f)
with open(os.path.join(args.data_path, "crosstask_release/activity_step.json"), "r") as f:
    act_cls = json.load(f)

##################################
# If using existing data-split   #
##################################
if args.exist_datasplit:
    with open("./checkpoints/CrossTask_t{}_datasplit_pre.pth".format(args.max_traj_len), "rb") as f:
        datasplit = pickle.load(f)
    trainset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
    )
    testset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
        train=False,
    )
    trainset.plan_vids = datasplit["train"]
    testset.plan_vids = datasplit["test"]

else:
    """ Random Split dataset by video """
    train_vids, test_vids = random_split(
        task_vids, test_tasks, args.n_train, seed=args.seed)

    trainset = CrossTaskDataset(
        train_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls
    )

    # Run random_split for eval/test sub-set
    # trainset.random_split()
    testset = CrossTaskDataset(
        test_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
        train=False
    )

#######################
# Run data whitening  #
#######################

mean_lang = 0.038948704
mean_vis = 0.000133333
var_lang = 33.063942
var_vis = 0.00021489676

trainset.mean_lan = mean_lang
trainset.mean_vis = mean_vis
trainset.var_lan = var_lang
trainset.var_vis = var_vis
testset.mean_lan = mean_lang
testset.mean_vis = mean_vis
testset.var_lan = var_lang
testset.var_vis = var_vis


#######################
# Init the DataLoader #
#######################
train_loader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True,
    # collate_fn=collate_func,
)
val_loader = DataLoader(
    testset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=True,
    # collate_fn=collate_func,
)
# Show stats of train/test dataset
print("Training dataset has {} samples".format(len(trainset)))
print("Testing dataset has {} samples".format(len(testset)))

"""Get all reference from test-set, for KL-Divgence, NLL, MC-Prec and MC-Rec"""
reference = [x[2] for x in testset.plan_vids]
all_ref = np.array(reference)

time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())

logname = 'exptower5_' + time_pre + '_' + str(args.dataset_mode) + '_' + str(args.max_traj_len)

args.logname = logname

##################################
# Saving the data split to local #
##################################
if not args.exist_datasplit:
    datasplit = {}

    datasplit["train"] = trainset.plan_vids
    datasplit["test"] = testset.plan_vids

    with open("CrossTask_t{}_datasplit.pth".format(args.max_traj_len), "wb") as f:
        pickle.dump(datasplit, f)

def main():
    global best_loss, best_acc, best_success_rate, best_miou

    # create model

    from model.model_baseline_tower5 import Model
    model = Model(args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, top1_acc {})"
              .format(args.resume, checkpoint['epoch'], checkpoint['best_top1_acc']))

    if args.start_epoch is None:
        args.start_epoch = 0

    model = model.cuda()
    cudnn.benchmark = True

    num_param = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: ', num_param)

    optimizer = torch.optim.SGD(model.parameters(),
                                momentum=args.momentum,
                                lr=args.lr,
                                weight_decay=args.weight_decay)

    criterion = FocalLoss(gamma=args.gamma)

    tb_logdir = os.path.join('./logs', logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    log, writer = set_save_path(tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, args.lr_steps)

        epoch_starttime = time.time()

        train_loss, train_state_loss, train_acc, train_success_rate, train_miou = train(args, train_loader, model, optimizer, epoch, criterion, tb_logger)
        loss, acc, success_rate, miou = validate(args, val_loader, model, criterion, epoch, tb_logger)

        epoch_endtime = time.time()

        oneepoch_time = epoch_endtime - epoch_starttime

        print('one epoch time:', oneepoch_time)

        print('t/T=', oneepoch_time * epoch, '/', oneepoch_time * args.epochs)

        print('SSSSSSSSSSSSSSETTING', args)

        is_best_sr = success_rate > best_success_rate
        if is_best_sr:
            best_loss = loss
            best_acc = acc
            best_success_rate = success_rate
            best_miou = miou
        print(
            'Epoch {}: Best evaluation - '
            'accuracy: {:.2f}, success rate: {:.2f}, miou: {:.2f}'
                .format(epoch, best_acc, best_success_rate, best_miou))
        if not os.path.exists(args.ckpt):
            os.makedirs(args.ckpt)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_success_rate': best_success_rate,
            },
            is_best_sr,  # is_best,
            os.path.join(tb_logdir, '{}'.format(logname)))

        log_info = ['epoch {}/{}'.format(epoch, args.epochs)]
        log_info.append('train: train_loss={:.4f}'.format(train_loss))
        log_info.append('train_acc={:.4f}'.format(train_acc))
        log_info.append('train_success_rate={:.4f}'.format(train_success_rate))
        log_info.append('train_MIoU={:.4f}'.format(train_miou))

        log_info.append('val: val_loss={:.4f}'.format(loss))
        log_info.append('val_acc={:.4f}'.format(acc))
        log_info.append('val_success_rate={:.4f}'.format(success_rate))
        log_info.append('val_MIoU={:.4f}'.format(miou))

        log_info.append('best: best_loss={:.4f}'.format(best_loss))
        log_info.append('best_acc={:.4f}'.format(best_acc))
        log_info.append('best_success_rate={:.4f}'.format(best_success_rate))
        log_info.append('best_MIoU={:.4f}'.format(best_miou))
        # writer.flush()
        log(', '.join(log_info))

        if epoch == 1:
            tb_logger.log_info(args)

def train(args, train_loader, model, optimizer, epoch, criterion, tb_logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    state_losses = AverageMeter()
    acc_meter = AverageMeter()
    success_rate_meter = AverageMeter()
    miou_meter = AverageMeter()

    model.train()
    end = time.time()

    for i, (_, _, frames, _, lowlevel_labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        frames = frames.cuda()
        lowlevel_labels = lowlevel_labels.cuda()

        if i == 0:
            print('lowlevel label pre', lowlevel_labels[0, :])


        output1, output2, output3, output = model(frames)

        output_reshaped = output.contiguous().view(-1, output.shape[-1])

        lowlevel_labels_reshaped = lowlevel_labels.contiguous().view(-1)

        loss4 = criterion(output_reshaped, lowlevel_labels_reshaped.long().cuda())

        lowlevel_labels1 = torch.cat([lowlevel_labels[:, 0:2], lowlevel_labels[:, 4:5]], dim=1)

        output_reshaped1 = output1.contiguous().view(-1, output1.shape[-1])

        lowlevel_labels_reshaped1 = lowlevel_labels1.contiguous().view(-1)

        loss1 = criterion(output_reshaped1, lowlevel_labels_reshaped1.long().cuda())

        lowlevel_labels2 = torch.cat([lowlevel_labels[:, 0:1], lowlevel_labels[:, 2:3],
                                      lowlevel_labels[:, 4:5]], dim=1)

        output_reshaped2 = output2.contiguous().view(-1, output2.shape[-1])

        lowlevel_labels_reshaped2 = lowlevel_labels2.contiguous().view(-1)

        loss2 = criterion(output_reshaped2, lowlevel_labels_reshaped2.long().cuda())

        lowlevel_labels3 = torch.cat([lowlevel_labels[:, 0:1], lowlevel_labels[:, 3:5]], dim=1)

        output_reshaped3 = output3.contiguous().view(-1, output3.shape[-1])

        lowlevel_labels_reshaped3 = lowlevel_labels3.contiguous().view(-1)

        loss3 = criterion(output_reshaped3, lowlevel_labels_reshaped3.long().cuda())

        loss = loss1 + loss2 + loss3 + loss4

        if i == 0:
            print('lowlevel label1 post', lowlevel_labels1[0, :])
            print('lowlevel label2 post', lowlevel_labels2[0, :])
            print('lowlevel label3 post', lowlevel_labels3[0, :])

        acc, success_rate, _ = accuracy(output_reshaped.cpu(), lowlevel_labels_reshaped.cpu(), args.max_traj_len)

        _, output_r = output.topk(1, 2, True, True)
        gt = output_r.squeeze(-1).cpu().numpy().astype("int")
        rst = lowlevel_labels.squeeze(-1).cpu().numpy().astype("int")
        miou = acc_iou(rst, gt, False)
        miou = miou.mean()

        losses.update(loss.item(), frames.size(0))

        acc_meter.update(acc.item(), frames.size(0))
        success_rate_meter.update(success_rate.item(), frames.size(0))
        miou_meter.update(miou, frames.size(0) // args.max_traj_len)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'State Loss {state_loss.val:.4f} ({state_loss.avg:.4f})\t'                  
                  'Train Acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t'
                  'Train Success Rate {success_rate_meter.val:.2f} ({success_rate_meter.avg:.2f})\t'
                  'Train_MIoU {miou_meter.val:.2f} ({miou_meter.avg:.2f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, state_loss=state_losses,
                acc_meter=acc_meter,
                success_rate_meter=success_rate_meter,
                miou_meter=miou_meter))

        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train/IterLoss'] = losses.val
            logs['Train/Acc'] = acc_meter.val
            logs['Train/Success_Rate'] = success_rate_meter.val
            logs['Train/MIoU'] = miou_meter.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()
    return losses.avg, state_losses.avg, acc_meter.avg, success_rate_meter.avg, miou_meter.avg

def validate(args, val_loader, model, criterion, epoch, tb_logger):

    batch_time = AverageMeter()
    losses = AverageMeter()
    state_losses = AverageMeter()
    acc_meter = AverageMeter()
    success_rate_meter = AverageMeter()
    miou_meter = AverageMeter()

    end = time.time()

    for i, (_, _, frames, _, lowlevel_labels, _) in enumerate(val_loader):

        frames = frames.cuda()
        lowlevel_labels = lowlevel_labels.cuda()
        with torch.no_grad():

            output1, output2, output3, output\
                = model(frames)

            output_reshaped = output.contiguous().view(-1, output.shape[-1])

            lowlevel_labels_reshaped = lowlevel_labels.contiguous().view(-1)

            loss4 = criterion(output_reshaped, lowlevel_labels_reshaped.long().cuda())

            lowlevel_labels1 = torch.cat([lowlevel_labels[:, 0:2], lowlevel_labels[:, 4:5]], dim=1)

            output_reshaped1 = output1.contiguous().view(-1, output1.shape[-1])

            lowlevel_labels_reshaped1 = lowlevel_labels1.contiguous().view(-1)

            loss1 = criterion(output_reshaped1, lowlevel_labels_reshaped1.long().cuda())

            lowlevel_labels2 = torch.cat(
                [lowlevel_labels[:, 0:1], lowlevel_labels[:, 2:3], lowlevel_labels[:, 4:5]], dim=1)

            output_reshaped2 = output2.contiguous().view(-1, output2.shape[-1])

            lowlevel_labels_reshaped2 = lowlevel_labels2.contiguous().view(-1)

            loss2 = criterion(output_reshaped2, lowlevel_labels_reshaped2.long().cuda())

            lowlevel_labels3 = torch.cat([lowlevel_labels[:, 0:1], lowlevel_labels[:, 3:5]], dim=1)

            output_reshaped3 = output3.contiguous().view(-1, output3.shape[-1])

            lowlevel_labels_reshaped3 = lowlevel_labels3.contiguous().view(-1)

            loss3 = criterion(output_reshaped3, lowlevel_labels_reshaped3.long().cuda())

            loss = loss1 + loss2 + loss3 + loss4

            acc, success_rate, _ = accuracy(output_reshaped.cpu(), lowlevel_labels_reshaped.cpu(), max_traj_len=args.max_traj_len)

            _, output_r = output.topk(1, 2, True, True)
            gt = output_r.squeeze(-1).cpu().numpy().astype("int")
            rst = lowlevel_labels.squeeze(-1).cpu().numpy().astype("int")
            miou = acc_iou(rst, gt, False)
            miou = miou.mean()

        losses.update(loss.item(), frames.size(0))
        acc_meter.update(acc.item(), frames.size(0))
        success_rate_meter.update(success_rate.item(), frames.size(0))
        miou_meter.update(miou, frames.size(0) // args.max_traj_len)

        batch_time.update(time.time() - end)

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Val Acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t'
                  'Val Success Rate {success_rate_meter.val:.2f} ({success_rate_meter.avg:.2f})\t'
                  'Val MIoU {miou_meter.val:.1f} ({miou_meter.avg:.2f})\t'
                .format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                acc_meter=acc_meter, success_rate_meter=success_rate_meter,
                miou_meter=miou_meter))

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val/EpochLoss'] = losses.avg
        logs['Val/Acc'] = acc_meter.val
        logs['Val/Success_Rate'] = success_rate_meter.val
        logs['Val/MIoU'] = miou_meter.val
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg, acc_meter.avg, success_rate_meter.avg, miou_meter.avg

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

def adjust_learning_rate(args, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, max_traj_len=0):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # Token Accuracy

        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)


        correct_1 = pred.eq(target.view(-1, 1))  # .view(-1, max_traj_len) # (bz, 1)
        # Instruction Accuracy
        instruction_correct = torch.all(correct_1, dim=1)
        instruction_accuracy = instruction_correct.sum() * 100.0 / instruction_correct.shape[0]

        # Success Rate
        trajectory_success = torch.all(instruction_correct.view(correct_1.shape[0] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        pred_inst = pred
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[0], -1)
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            # print(pred_inst[i], target_inst[i])
            pred_inst_set.add(tuple(pred_inst[i].tolist()))
            target_inst_set.add(tuple(target_inst[i].tolist()))
        MIoU = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(pred_inst_set.union(target_inst_set))
        return instruction_accuracy, trajectory_success_rate, MIoU

def acc_iou(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """

    epsn = 1e-6

    if aggregate:
        intersection = (pred & gt).sum((0, 1))
        union = (pred | gt).sum((0, 1))
    else:
        intersection = (pred & gt).sum((1))
        union = (pred | gt).sum((1))

    return 100 * ((intersection + epsn) / (union + epsn))

if __name__ == '__main__':
    main()