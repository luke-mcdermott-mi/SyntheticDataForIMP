import os
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from train import define_model, train
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
import models.resnet as RN
import models.densenet_cifar as DN
from coreset import randomselect, herding
from efficientnet_pytorch import EfficientNet
import models.convnet as CN
from train import validate
import torch.nn.utils.prune as prune
import copy


DATA_PATH = "./results"

def return_data_path(args):
    if args.factor > 1:
        init = 'mix'
    else:
        init = 'random'
    if args.dataset == 'imagenet' and args.nclass == 100:
        args.slct_type = 'idc_cat'
        args.nclass_sub = 20

    if 'idc' in args.slct_type:
        name = args.name
        if name == '':
            if args.dataset == 'cifar10':
                name = f'cifar10/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'cifar100':
                name = f'cifar100/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'imagenet':
                if args.nclass == 10:
                    name = f'imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor{args.factor}_{init}'
                elif args.nclass == 100:
                    name = f'imagenet100/resnet10apin_grad_l1_pt5_nd500_cut_nlr0.1_wd0.0001_factor{args.factor}_lr0.001_b_real128_{init}'

            elif args.dataset == 'svhn':
                name = f'svhn/conv3in_grad_mse_nd500_cut_niter2000_factor{args.factor}_lr0.005_{init}'
                if args.factor == 1 and args.ipc == 1:
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_cutout_scale_rotate'

            elif args.dataset == 'mnist':
                if args.factor == 1:
                    name = f'mnist/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'
                else:
                    name = f'mnist/conv3in_grad_l1_nd500_niter2000_factor{args.factor}_color_crop_lr0.0001_{init}'
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_scale_rotate'

            elif args.dataset == 'fashion':
                name = f'fashion/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'

        path_list = [f'{name}_ipc{args.ipc}']

    elif args.slct_type == 'dsa':
        path_list = [f'cifar10/dsa/res_DSA_CIFAR10_ConvNet_{args.ipc}ipc']
    elif args.slct_type == 'kip':
        path_list = [f'cifar10/kip/kip_ipc{args.ipc}']
    else:
        path_list = ['']

    return path_list

def convnet3(args, nclass, logger=None):
    width = 128
    model = CN.ConvNet(nclass,
                           net_norm=args.norm_type,
                           net_depth=args.depth,
                           net_width=width,
                           channel=args.nch,
                           im_size=(args.size, args.size))
    return model

def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model


def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    save_img('./results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec


def load_data_path(args):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    print()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if 'idc' in args.slct_type:
            if args.slct_type == 'idc':
                data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))

            elif args.slct_type == 'idc_cat':
                data_all = []
                target_all = []
                for idx in range(args.nclass // args.nclass_sub):
                    path = f'{args.save_dir}_{args.nclass_sub}_phase{idx}'
                    data, target = torch.load(os.path.join(path, 'data.pt'))
                    data_all.append(data)
                    target_all.append(target)
                    print(f"Load data from {path}")

                data = torch.cat(data_all)
                target = torch.cat(target_all)

            print("Load condensed data ", data.shape, args.save_dir)

            if args.factor > 1:
                data, target = decode(args, data, target)
            train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
            train_dataset = TensorDataset(data, target, train_transform)
        else:
            train_dataset = ImageFolder(traindir,
                                        train_transform,
                                        nclass=args.nclass,
                                        seed=args.dseed,
                                        slct_type=args.slct_type,
                                        ipc=args.ipc,
                                        load_memory=args.load_memory)
            print(f"Test {args.dataset} random selection {args.ipc} (total {len(train_dataset)})")
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)

    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        elif args.dataset == 'svhn':
            transform_fn = transform_svhn
        elif args.dataset == 'mnist':
            transform_fn = transform_mnist
        elif args.dataset == 'fashion':
            transform_fn = transform_fashion
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

        # Load condensed dataset
        if 'idc' in args.slct_type:
            data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
            print("Load condensed data ", args.save_dir, data.shape)
            # This does not make difference to the performance
            # data = torch.clamp(data, min=0., max=1.)
            if args.factor > 1:
                data, target = decode(args, data, target)

            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)

        elif args.slct_type in ['dsa', 'kip']:
            condensed = torch.load(f'{args.save_dir}.pt')
            try:
                condensed = condensed['data']
                data = condensed[-1][0]
                target = condensed[-1][1]
            except:
                data = condensed[0].permute(0, 3, 1, 2)
                target = torch.arange(args.nclass).repeat_interleave(len(data) // args.nclass)

            if args.factor > 1:
                data, target = decode(args, data, target)
            # These data are saved as the normalized values!
            train_transform, _ = transform_fn(augment=args.augment,
                                              from_tensor=True,
                                              normalize=False)
            train_dataset = TensorDataset(data, target, train_transform)
            print("Load condensed data ", args.save_dir, data.shape)

        else:
            if args.dataset == 'cifar10':
                train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                             train=True,
                                                             transform=train_transform,
                                                             download=True)
            elif args.dataset == 'cifar100':
                train_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                              train=True,
                                                              transform=train_transform,
                                                             download=True)
            elif args.dataset == 'svhn':
                train_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                          split='train',
                                                          transform=train_transform)
                train_dataset.targets = train_dataset.labels
            elif args.dataset == 'mnist':
                train_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                           train=True,
                                                           transform=train_transform)
            elif args.dataset == 'fashion':
                train_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                                  train=True,
                                                                  transform=train_transform)

            indices = randomselect(train_dataset, args.ipc, nclass=args.nclass)
            train_dataset = Subset(train_dataset, indices)
            print(f"Random select {args.ipc} data (total {len(indices)})")

        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       transform=test_transform,
                                                        download=True)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        transform=test_transform,
                                                        download=True)
        elif args.dataset == 'svhn':
            val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                    split='test',
                                                    transform=test_transform)
        elif args.dataset == 'mnist':
            val_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                     train=False,
                                                     transform=test_transform)
        elif args.dataset == 'fashion':
            val_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                            train=False,
                                                            transform=test_transform)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    os.makedirs('./results', exist_ok=True)
    save_img('./results/test.png',
             torch.stack([d[0] for d in train_dataset]),
             dataname=args.dataset)
    print()

    return train_dataset, val_dataset

def get_parameters_to_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)
        
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


def test_interpolation(args, logger, train_loader, path, model_fn):
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()

    nclass = args.nclass
    np.random.seed(4)
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)

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
    
    n_samples = 10000 #global samples
    num_nearby = 40 #explore near trained models more
    results = np.zeros((n_samples, 4))

    for i in range(n_samples):
        temp = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
        X = torch.rand(1).item()*4-2 #random number from -2,2
        Y = torch.rand(1).item()*4-2
        direction = get_linear_combination(V[0], V[1], temp, X, Y)

        temp = model_fn(args, nclass, logger=logger)
        prune.global_unstructured(get_parameters_to_prune(temp),pruning_method=prune.L1Unstructured,amount=0)
        I = get_linear_combination(direction, M, temp, 1 , 1)
        top1, top5, loss = validate(args, train_loader, I, criterion, i, logger=None)
        print(X,Y, top1, loss)
        results[i] = [-X-Y, X-Y, loss, top1]
        np.save(path + 'overnight_loss_landscape_imp_distdata' + str(prune_iter),results)
    
    counter = n_samples


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






if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    import numpy as np
    cudnn.benchmark = True

    device = args.device
    torch.cuda.set_device(device)

    if args.same_compute and args.factor > 1:
        args.epochs = int(args.epochs / args.factor**2)

    path_list = return_data_path(args)
    for p in path_list:
        args.save_dir = os.path.join(DATA_PATH, p)
        if args.slct_type == 'herding':
            train_dataset, val_dataset = herding(args)
        else:
            train_dataset, val_dataset = load_data_path(args)

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        val_loader = MultiEpochsDataLoader(val_dataset,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
        
        print('Convnet3 Loss Landscape!')
        save_path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/convnet-3/'
        test_interpolation(args, None, train_loader, save_path, convnet3)
        
        print('ResNet-10 Loss Landscape!')
        save_path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/resnet-10/'
        test_interpolation(args, None, train_loader, save_path, resnet10_bn)
        
        print('ResNet-18 Loss Landscape!')
        save_path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/resnet-18/'
        test_interpolation(args, None, train_loader, save_path, resnet18_bn)
        

