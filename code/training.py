import argparse
import os
import random
import datetime
import numpy as np
from math import sqrt
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from skimage import io
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter
from datasets import Levir_CD as Data

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--encoder', required=True, choices=['ResNet', 'SAM', 'effSAM']) 
parser.add_argument('--train_batch_size', required=False, default=128, type=int)
parser.add_argument('--val_batch_size', required=False, default=32, type=int)
parser.add_argument('--lr', required=False, default=0.1, type=float)
parser.add_argument('--epochs', required=False, default=100, type=int)
parser.add_argument('--gpu', required=False, default=True, action='store_true')
parser.add_argument('--dev_id', required=False, default=0, type=int)
parser.add_argument('--multi_gpu', required=False, default=None, type=str)
parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)
parser.add_argument('--weight_decay', required=False, default=5e-4, type=float)
parser.add_argument('--momentum', required=False, default=0.9, type=float)
parser.add_argument('--print_freq', required=False, default=200, type=int)
parser.add_argument('--predict_step', required=False, default=5, type=int)
parser.add_argument('--crop_size', required=False, default=512, type=int)
parser.add_argument('--chkpt_dir', required=False, default='../results/checkpoints')
args = parser.parse_args()
encoder = args.encoder
 


def main():
    
    if encoder == 'ResNet':
        from models.ResNet_CD import ResNet_CD as Net
    elif encoder == 'SAM':
        from models.SAM_CD import SAM_CD as Net
    elif encoder == 'effSAM':
        from models.effSAM_CD import SAM_CD as Net
    else:
        raise Exception(f'Unknown encoder {encoder}')
    
    net = Net()
    if args.multi_gpu:
        net = torch.nn.DataParallel(net, [int(id) for id in args.multi_gpu.split(',')])
    net.to(device=torch.device('cuda', int(args.dev_id)))

    train_set = Data.RS('train', random_crop=True, crop_nums=10, crop_size=args.crop_size, random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.data_loader_num_workers, shuffle=True)
    val_set = Data.RS('val', sliding_crop=False, crop_size=args.crop_size, random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=args.data_loader_num_workers, shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    print(f'Training {encoder} started')
    train(train_loader, net, optimizer, val_loader)
    print(f'Training {encoder} finished')


def train(train_loader, net, optimizer, val_loader):
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    current_time = time.localtime(begin_time)
    date_str = time.strftime("%d-%m-%Y_%H-%M", current_time)
    all_iters = float(len(train_loader) * args.epochs)
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, labels = data
            if args.gpu:
                imgs_A = imgs_A.to(torch.device('cuda', int(args.dev_id))).float()
                imgs_B = imgs_B.to(torch.device('cuda', int(args.dev_id))).float()
                labels = labels.to(torch.device('cuda', int(args.dev_id))).float().unsqueeze(1)
            else:  # CPU
                labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = net(imgs_A, imgs_B)  

            assert outputs.shape[1] == 1
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args.print_freq == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                loss_rec = train_loss.val

        val_F, val_acc, val_IoU, val_loss = validate(val_loader, net, curr_epoch)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            # torch.save(net.state_dict(), os.path.join(args.chkpt_dir, NET_NAME + '_e%d_OA%.2f_F%.2f_IoU%.2f.pth' % (
            # curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
            torch.save(net.state_dict(), os.path.join(args.chkpt_dir, 
            f"{encoder}_e{curr_epoch}_OA{val_acc * 100:.2f}_F{val_F * 100:.2f}_IoU{val_IoU * 100:.2f}_{date_str}.pth"))
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args.epochs, time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                 bestIoU * 100))
        curr_epoch += 1
        if curr_epoch >= args.epochs:
            return

def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data

        if args.gpu:
            imgs_A = imgs_A.to(torch.device('cuda', int(args.dev_id))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args.dev_id))).float()
            labels = labels.to(torch.device('cuda', int(args.dev_id))).float().unsqueeze(1)
        else:  # CPU
            labels = labels.float().unsqueeze(1)

        with torch.no_grad():
            outputs = net(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100))


    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args.lr * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr

if __name__ == '__main__':
    main()