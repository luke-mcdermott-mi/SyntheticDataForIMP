# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
from misc.augment import DiffAug
from efficientnet_pytorch import EfficientNet
import time
import warnings
from prune import get_parameters_to_prune, convnet3, resnet10_bn, resnet18_bn
import torch.nn.utils.prune as prune
from torch.autograd import grad

warnings.filterwarnings("ignore")
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

mean_torch = {}
std_torch = {}
for key, val in MEANS.items():
    mean_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)
for key, val in STDS.items():
    std_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)




def calc_hessian(args, logger):
    cudnn.benchmark = True
    _, train_loader, val_loader, nclass = load_data(args)

    if args.model == 'convnet3':
        model_fn = convnet3
        if args.dataset == 'cifar10':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/convnet-3/'
        elif args.dataset == 'cifar100':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_100/convnet-3/'    
        elif args.dataset == 'imagenet' and args.nclass == 10:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_10/convnet-3/'    
        elif args.dataset == 'imagenet' and args.nclass == 100:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_100/convnet-3/'
    elif args.model == 'resnet10_bn':
        model_fn = resnet10_bn
        if args.dataset == 'cifar10':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/resnet-10/'
        elif args.dataset == 'cifar100':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_100/resnet-10/'    
        elif args.dataset == 'imagenet' and args.nclass == 10:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_10/resnet-10/'    
        elif args.dataset == 'imagenet' and args.nclass == 100:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_100/resnet-10/'
    elif args.model == 'resnet18_bn':
        model_fn = resnet18_bn
        if args.dataset == 'cifar10':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/resnet-18/'
        elif args.dataset == 'cifar100':
            path = '/home/ubuntu/luke/LMC/pruned_models/cifar_100/resnet-18/'    
        elif args.dataset == 'imagenet' and args.nclass == 10:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_10/resnet-18/'    
        elif args.dataset == 'imagenet' and args.nclass == 100:
            path = '/home/ubuntu/luke/LMC/pruned_models/imagenet_100/resnet-18/'
    else:
        print('No model found')
    
    np.random.seed(3)
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    prune_iter = args.prune_iter
    prune_method = 'syn'

    model = model_fn(args, nclass, logger=logger)
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    if prune_method == 'imp':
        model.load_state_dict(torch.load(path + 'seed0_iter' + str(prune_iter) + '_imp'))
    else:
        model.load_state_dict(torch.load(path + 'seed0_iter' + str(prune_iter)))
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    model = model.to('cuda:' + str(args.device)) 
    hes_values = compute_hessian_diagonal(model, train_loader)
    print(hes_values)

    torch.save(hes_values, path + 'hesv_' + str(prune_method) + '_iter' + str(prune_iter))


def compute_hessian_diagonal(model, dataloader):
    # Initialize loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use a single mini-batch of data
    x_batch, y_batch = next(iter(dataloader))
    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()
    
    # Forward pass to compute outputs
    outputs = model(x_batch)
    print('compute loss')
    # Compute loss
    loss = criterion(outputs, y_batch)
    print('compute first order grad')
    # Compute first-order gradients
    grads = grad(loss, model.parameters(), create_graph=True)
    print('compute hessian diagonal')
    # Initialize Hessian diagonal
    hessian_diag = torch.zeros(sum(p.numel() for p in model.parameters())).cuda()
    
    # Compute diagonal of Hessian approximation
    grad_idx = 0
    for g in grads:
        g = g.flatten()
        for i in range(len(g)):
            grad2 = grad(g[i], model.parameters(), retain_graph=True)
            hessian_diag_component = torch.cat([g_.reshape(-1) for g_ in grad2])
            hessian_diag[grad_idx] = hessian_diag_component[grad_idx].item()
            grad_idx += 1
            
    return hessian_diag

# Assuming model and dataloader are defined
# hessian_diag = compute_hessian_diagonal(model, dataloader)


def compute_sorted_eigenvalues(model, dataloader):
    # Initialize Hessian as zero matrix
    num_params = sum(p.numel() for p in model.parameters())
    hessian_approx = torch.zeros((num_params, num_params))
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Use a single mini-batch of data to compute Hessian approximation
    x_batch, y_batch = next(iter(dataloader))
    x_batch = x_batch.to('cuda:3')
    y_batch = y_batch.to('cuda:3')
    x_batch.requires_grad = True
    
    print('Forward Pass')
    # Forward pass to compute outputs
    outputs = model(x_batch)
    
    print('Compute loss')
    # Compute loss using the appropriate criterion
    loss = criterion(outputs, y_batch)
    
    print('Get Gradients')
    # Compute first-order gradients
    grads = grad(loss, model.parameters(), create_graph=True)
    
    # Flatten gradients
    grads_flat = torch.cat([g.reshape(-1) for g in grads])
    
    print('Compute Hessian')
    # Compute Hessian approximation for the mini-batch
    for i in range(len(grads_flat)):
        grad2 = grad(grads_flat[i], model.parameters(), retain_graph=True)
        hessian_row = torch.cat([g.reshape(-1) for g in grad2])
        hessian_approx[i] = hessian_row
    
    print('Compute Eigenvalues')
    # Compute eigenvalues of the approximated Hessian matrix
    eigenvalues = torch.linalg.eigvals(hessian_approx).real
    
    # Sort eigenvalues in ascending order
    sorted_eigenvalues = torch.sort(eigenvalues)[0]
    return sorted_eigenvalues

            




def train(args, model, train_loader, val_loader, plotter=None, logger=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
        # TODO: optimizer scheduler steps

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    # print(get_time())
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch,
                                          logger,
                                          aug,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

            if plotter != None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None:
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                logger=None,
                aug=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    num_exp = 0
    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()

        data_time.update(time.time() - end)

        if aug != None:
            with torch.no_grad():
                input = aug(input)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None):
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def validate(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(val_loader),
                                                                     batch_time=batch_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

    if logger is not None:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
            path, cur_epoch, checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, state, is_best):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    print("checkpoint saved! ", ckpt_path)


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    device = args.device
    torch.cuda.set_device(device)

    calc_hessian(args, logger)
