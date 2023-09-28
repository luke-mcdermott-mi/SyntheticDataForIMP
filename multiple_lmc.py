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


def sparsity_print(model):
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    zero = total = 0
    for module, _ in get_parameters_to_prune(model):
        zero += float(torch.sum(module.weight == 0))
        total += float(module.weight.nelement())
    print('Number of Zero Weights:', zero)
    print('Total Number of Weights:', total)
    print('Sparsity', zero/total)
    return zero, total


def test_interpolation(args, logger):
    cudnn.benchmark = True
    _, train_loader, val_loader, nclass = load_data(args)
    criterion = nn.CrossEntropyLoss().cuda()
    print(len(train_loader))
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
        
    

    np.random.seed(4)
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)
    """
    all_seeds = [0,1,2,3,4,5,6,7,8]
    all_combinations = []
    for x in range(9):
        for y in range(x+1,9):
            all_combinations.append((x,y))

    prune_iter = args.prune_iter

    results = np.zeros((36,2,11)) #Combinations x (Loss or Acc) x Interpolated Values
    
    for i, (a,b) in enumerate(all_combinations):

        #Load the correct endpoint models
        A = model_fn(args, nclass, logger=logger)
        B = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(A),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(B),pruning_method=prune.L1Unstructured,amount=0)
        A.load_state_dict(torch.load(path + 'seed' + str(a) + '_iter' + str(prune_iter)))
        B.load_state_dict(torch.load(path + 'seed' + str(b) + '_iter' + str(prune_iter)))
        prune.global_unstructured(get_parameters_to_prune(A),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(B),pruning_method=prune.L1Unstructured,amount=0)
        
        #Generate 11 interpolated models per combination
        for idx, beta in enumerate(np.arange(0,1.1,.1)):
            temp_model = model_fn(args, nclass, logger=logger)
            prune.global_unstructured(get_parameters_to_prune(temp_model),pruning_method=prune.L1Unstructured,amount=0)

            interpolated_model = get_interpolated_model(A, B, temp_model, beta)
            top1, top5, loss = validate(args, train_loader, interpolated_model, criterion, beta, logger=None)
            results[i, 0, idx] = loss
            results[i, 1, idx] = top1
            print('Iter: ', prune_iter, ', Beta: ', beta, ', Loss: ', loss, ', Acc: ', top1)

        np.save(path + 'multiple_standard_interp' + str(prune_iter),results)
    """
    #Set up a square loss landscape, with M as middle of the square, and V as perpendicular vectors.
    prune_iter = args.prune_iter
    base_models = [model_fn(args, nclass, logger=logger) for i in range(4)]
    _ = [prune.global_unstructured(get_parameters_to_prune(m),pruning_method=prune.L1Unstructured,amount=0) for m in base_models]
    _ = [m.load_state_dict(torch.load(path + 'seed' + str(i) + '_iter' + str(prune_iter) + '_imp')) for i,m in enumerate(base_models)]
    _ = [prune.global_unstructured(get_parameters_to_prune(m),pruning_method=prune.L1Unstructured,amount=0) for m in base_models]
    base_locs = [(-1,1), (1,1), (1,-1), (-1,-1)]

    temp = model_fn(args, nclass, logger=logger)
    prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
    M = get_interpolated_model(base_models[0], base_models[1], temp, .5) #middle model of square
    temp1 = model_fn(args, nclass, logger=logger)
    temp2 = model_fn(args, nclass, logger=logger)
    prune.global_unstructured(get_parameters_to_prune(temp1),pruning_method=prune.L1Unstructured,amount=0)
    prune.global_unstructured(get_parameters_to_prune(temp2),pruning_method=prune.L1Unstructured,amount=0)
    V = [
        get_linear_combination(base_models[0], base_models[1], temp1, .5, -.5),
        get_linear_combination(base_models[3], base_models[2], temp2, .5, -.5)
    ]

    #end testing
    
    n_samples = 10000 #global samples
    num_nearby = 40 #explore near trained models more
    results = np.zeros((n_samples, 4))
    """
    if not os.path.exists(path + 'loss_landscape' + str(prune_iter)+ '.npy'):
        print('not found')
        results = np.zeros((66 + n_samples + num_nearby*4, 4))
    else:
        print('found')
        old_results = np.load(path + 'loss_landscape' + str(prune_iter) + '.npy')
        results = np.zeros((len(old_results) + 66 + n_samples + num_nearby*4, 4))
        results[:len(old_results)] = old_results
    """
    xs = np.linspace(0.0007970690550475478, 0.0007971464856703357, 100)
    ys = np.linspace(0.0010145217103828756, 0.0010146202652663821, 100)
    for i in range(n_samples):
        temp = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
        #X = (torch.rand(1).item()-.5)/128 #random number from -2,2
        #Y = (torch.rand(1).item()-.5)/128
        X = xs[i]
        Y = ys[i]
        direction = get_linear_combination(V[0], V[1], temp, X, Y)

        temp = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
        I = get_linear_combination(direction, base_models[0], temp, 1 , 1) #base_models[0] used to be M
        top1, top5, loss = validate(args, train_loader, I, criterion, i, logger=None)
        print(X,Y, top1, loss)
        results[i] = [-X-Y, X-Y, loss, top1]
        np.save(path + 'single_losslandscape_imp_' + str(prune_iter),results)
    
    counter = n_samples
    """
    #0->1, 0->2, 0->3, 1->2, 1->3, 2->3
    #b,c,d,c,d,d
    print('Linear Interpolations')
    for a in range(4):
        for b in range(a+1,4):
            for beta in np.arange(0,1.1,.1):
                loc = (base_locs[a][0] * (1-beta) + base_locs[b][0]*beta, 
                       base_locs[a][1] * (1-beta) + base_locs[b][1]*beta)
                
                temp = model_fn(args, nclass, logger=logger)
                prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)

                I = get_interpolated_model(base_models[a], base_models[b], temp, beta)
                top1, top5, loss = validate(args, train_loader, I, criterion, 0, logger=None)
                print(loc, top1, loss)
                results[counter] = [loc[0], loc[1], loss, top1]
                counter += 1
                np.save(path + 'loss_landscape' + str(prune_iter),results)
    """
    """
    for s in range(4):
        print('S: ', s)
        for i in range(num_nearby):
            temp = model_fn(args, nclass, logger=logger)
            prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
            X = torch.rand(1).item()*1.5-.75 #random number from -.5,.5
            Y = torch.rand(1).item()*1.5-.75
            direction = get_linear_combination(V[0], V[1], temp, X, Y)

            temp = model_fn(args, nclass, logger=logger)
            prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
            I = get_linear_combination(direction, base_models[s], temp, 1 , 1)
            top1, top5, loss = validate(args, train_loader, I, criterion, i, logger=None)
            print(X,Y, top1, loss)
            results[counter] = [base_locs[s][0]-X-Y, base_locs[s][1]+X-Y, loss, top1]
            counter += 1
            np.save(path + 'loss_landscape' + str(prune_iter),results)
    """



    
    """
    results = np.zeros((3,2,11)) # x (Loss or Acc) x Interpolated Values
    
    for i, (a,b) in enumerate(all_combinations):
        seed0_model = model_fn(args, nclass, logger=logger)
        seed1_model = model_fn(args, nclass, logger=logger)
        #if prune_iter > 0:
        prune.global_unstructured(get_parameters_to_prune(seed0_model),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(seed1_model),pruning_method=prune.L1Unstructured,amount=0)

        seed0_model.load_state_dict(torch.load(path + 'seed' + str(a) + '_iter' + str(prune_iter)))
        seed1_model.load_state_dict(torch.load(path + 'seed' + str(b) + '_iter' + str(prune_iter)))

        prune.global_unstructured(get_parameters_to_prune(seed0_model),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(seed1_model),pruning_method=prune.L1Unstructured,amount=0)
        
        temp_model = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(temp_model),pruning_method=prune.L1Unstructured,amount=0)
        model_i = get_interpolated_model(seed0_model, seed1_model, temp_model, .5)

        print('Got model i')
        
        for seed in all_seeds:
            if not seed in [a,b]:
                #Get other model
                other_seed_model = model_fn(args, nclass, logger=logger)
                prune.global_unstructured(get_parameters_to_prune(other_seed_model),pruning_method=prune.L1Unstructured,amount=0)
                other_seed_model.load_state_dict(torch.load(path + 'seed' + str(seed) + '_iter' + str(prune_iter)))
                prune.global_unstructured(get_parameters_to_prune(other_seed_model),pruning_method=prune.L1Unstructured,amount=0)

                #interp other model with model i (model i is alreadty an interpolation)
                for idx, beta in enumerate(np.arange(0,1.1,.1)):
                    temporary_model = model_fn(args, nclass, logger=logger)
                    prune.global_unstructured(get_parameters_to_prune(temporary_model),pruning_method=prune.L1Unstructured,amount=0)

                    interpolated_model = get_interpolated_model(model_i, other_seed_model, temporary_model, beta)
                    top1, top5, loss = validate(args, val_loader, interpolated_model, criterion, beta, logger=None)
                    results[i, 0, idx] = loss
                    results[i, 1, idx] = top1
                    print('Iter: ', prune_iter, ', Beta: ', beta, ', Loss: ', loss, ', Acc: ', top1)

        np.save(path + 'multiple_modeli_interp' + str(prune_iter),results)
    """

def get_linear_combination(A, B, model, alpha, beta):
    model.eval()
    A.eval()
    B.eval()
    model = model.to(torch.device('cpu'))
    A = A.to(torch.device('cpu'))
    B = B.to(torch.device('cpu'))
    #print([name for name, module in model.named_modules()])
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]
            weight_interpolation = A_module.weight_orig.data * (alpha) + B_module.weight_orig.data * beta
            module.weight_orig.data.copy_(weight_interpolation)
            if module.bias is not None:
                bias_interpolation = A_module.bias.data * (alpha) + B_module.bias.data * beta
                module.bias.data.copy_(bias_interpolation)
        elif isinstance(module, torch.nn.BatchNorm2d):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]

            running_mean_interpolation = A_module.running_mean * (alpha) + B_module.running_mean * beta
            running_var_interpolation = A_module.running_var * (alpha) + B_module.running_var * beta

            module.running_mean.data.copy_(running_mean_interpolation)
            module.running_var.data.copy_(running_var_interpolation)

            # Interpolate batch norm weights and biases
            weight_interpolation = A_module.weight * (alpha) + B_module.weight * beta
            bias_interpolation = A_module.bias * (alpha) + B_module.bias * beta
            module.weight.data.copy_(weight_interpolation)
            module.bias.data.copy_(bias_interpolation)
        elif isinstance(module, torch.nn.GroupNorm):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]

            num_groups = module.num_groups  # Get the number of groups for GroupNorm

            # Interpolate parameters based on num_groups
            weight_interpolation = A_module.weight * (alpha) + B_module.weight * beta
            bias_interpolation = A_module.bias * (alpha) + B_module.bias * beta

            # Apply interpolated parameters to the current model's GroupNorm layer
            module.weight.data.copy_(weight_interpolation)
            module.bias.data.copy_(bias_interpolation)
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)

    #model = model.to(torch.device('cuda:' + str(args.device)))
    model = model.to(torch.device('cuda:' + str(args.device)))
    return model


def get_interpolated_model(A, B, model, beta):
    model.eval()
    A.eval()
    B.eval()
    model = model.to(torch.device('cpu'))
    A = A.to(torch.device('cpu'))
    B = B.to(torch.device('cpu'))
    #print([name for name, module in model.named_modules()])
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]
            weight_interpolation = A_module.weight_orig.data * (1 - beta) + B_module.weight_orig.data * beta
            module.weight_orig.data.copy_(weight_interpolation)
            if module.bias is not None:
                bias_interpolation = A_module.bias.data * (1 - beta) + B_module.bias.data * beta
                module.bias.data.copy_(bias_interpolation)
        elif isinstance(module, torch.nn.BatchNorm2d):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]

            running_mean_interpolation = A_module.running_mean * (1 - beta) + B_module.running_mean * beta
            running_var_interpolation = A_module.running_var * (1 - beta) + B_module.running_var * beta

            module.running_mean.data.copy_(running_mean_interpolation)
            module.running_var.data.copy_(running_var_interpolation)

            # Interpolate batch norm weights and biases
            weight_interpolation = A_module.weight * (1 - beta) + B_module.weight * beta
            bias_interpolation = A_module.bias * (1 - beta) + B_module.bias * beta
            module.weight.data.copy_(weight_interpolation)
            module.bias.data.copy_(bias_interpolation)
        elif isinstance(module, torch.nn.GroupNorm):
            A_module = list(A.named_modules())[idx][1]
            B_module = list(B.named_modules())[idx][1]

            num_groups = module.num_groups  # Get the number of groups for GroupNorm

            # Interpolate parameters based on num_groups
            weight_interpolation = A_module.weight * (1 - beta) + B_module.weight * beta
            bias_interpolation = A_module.bias * (1 - beta) + B_module.bias * beta

            # Apply interpolated parameters to the current model's GroupNorm layer
            module.weight.data.copy_(weight_interpolation)
            module.bias.data.copy_(bias_interpolation)
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)

    #model = model.to(torch.device('cuda:' + str(args.device)))
    model = model.to(torch.device('cuda:' + str(args.device)))
    return model

def compare_models(model, A):
    model_state_dict = model.state_dict()
    A_state_dict = A.state_dict()

    print("Comparing models...")
    for key in model_state_dict.keys():
        if key in A_state_dict:
            if not torch.equal(model_state_dict[key], A_state_dict[key]):
                print(f"Difference found in '{key}'")
        else:
            print(f"'{key}' not found in model A")

    for key in A_state_dict.keys():
        if key not in model_state_dict:
            print(f"'{key}' not found in current model")

    print("Comparison completed.")






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
    torch.cuda.set_device(args.device)
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

    test_interpolation(args, logger)
