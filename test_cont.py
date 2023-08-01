import torch.utils.data
from callbacks import AverageMeter
import time
from utils import *
from torch.utils.data import DataLoader
import json
import pickle
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
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--max_traj_len', default=3, type=int, help='action number')
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--dataset_root', default='./crosstask/')
parser.add_argument('--frameduration', default=3, type=int)
parser.add_argument('--num_workers', default=8, type=int)
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
parser.add_argument('--memory_size', default=128)
parser.add_argument('--N', default=1, type=int,help='Number of layers in the temporal decoder')
parser.add_argument('--H', default=16, type=int,help='Number of heads in the temporal decoder')
parser.add_argument('--d_model', default=1024, type=int)
parser.add_argument('--decoder_dropout', default=0, type=float)
parser.add_argument('--feat_dropout', default=0, type=float)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=999999999, type=int)

parser.add_argument('--exist_datasplit', default=True, type=bool)
parser.add_argument('--dim_feedforward', default=1024, type=int)
parser.add_argument('--mlp_mid', default=512, type=int)
parser.add_argument('--feat_mid', default=1024, type=int)
parser.add_argument('--query_length', default=4, type=int)
parser.add_argument('--memory_length', default=4, type=int)
parser.add_argument('--init_weight', default=True, type=bool)
parser.add_argument('--gamma', default=1.5, type=float)
parser.add_argument('--smallmid_ratio', default=3, type=int)
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

with open("./checkpoints/CrossTask_t{}_datasplit.pth".format(args.max_traj_len), "rb") as f:
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

from model.model_baseline_cont import Model
model = Model(args)
model = model.cuda()

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

def inference(args, model_path=False):
    # global args

    if model_path:
        model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        print("loading model weights from {}".format(model_path))
    # model.eval()

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
            output = model(frames)

            output_reshaped = output.contiguous().view(-1, output.shape[-1])

            lowlevel_labels_reshaped = lowlevel_labels.contiguous().view(-1)

            acc, success_rate, _ = accuracy(output_reshaped.cpu(), lowlevel_labels_reshaped.cpu(),
                                            max_traj_len=args.max_traj_len)

            _, output_r = output.topk(1, 2, True, True)
            gt = output_r.squeeze(-1).cpu().numpy().astype("int")
            rst = lowlevel_labels.squeeze(-1).cpu().numpy().astype("int")
            miou = acc_iou(rst, gt, False)
            miou = miou.mean()


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

    return losses.avg, acc_meter.avg, success_rate_meter.avg, miou_meter.avg


if __name__ == '__main__':
    inference(args, model_path='./checkpoints/CrossTask_t3_best.pth.tar')