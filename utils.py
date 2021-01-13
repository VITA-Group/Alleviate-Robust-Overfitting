import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd
from datasets import *
from models.preactivate_resnet import *
from models.vgg import *
from models.wideresnet import *

__all__ = ['save_checkpoint', 'setup_dataset_models', 'setup_dataset_models_standard', 'setup_seed', 'moving_average', 'bn_update', 'print_args',
            'train_epoch', 'train_epoch_adv', 'train_epoch_adv_dual_teacher',
            'test', 'test_adv']

def save_checkpoint(state, is_SA_best, is_RA_best, is_SA_best_swa, is_RA_best_swa, save_path, filename='checkpoint.pth.tar'):
    
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if is_SA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_SA_best.pth.tar'))
    if is_RA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_RA_best.pth.tar'))
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if is_RA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))

#print training configuration
def print_args(args):
    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    if args.arch == 'wideresnet':
        print('Depth {}'.format(args.depth_factor))
        print('Width {}'.format(args.width_factor))
    print('*'*50)        
    print('Attack Norm {}'.format(args.norm))  
    print('Test Epsilon {}'.format(args.test_eps))
    print('Test Steps {}'.format(args.test_step))
    print('Train Steps Size {}'.format(args.test_gamma))
    print('Test Randinit {}'.format(args.test_randinit))
    if args.eval:
        print('Evaluation')
        print('Loading weight {}'.format(args.pretrained))
    else:
        print('Training')
        print('Train Epsilon {}'.format(args.train_eps))
        print('Train Steps {}'.format(args.train_step))
        print('Train Steps Size {}'.format(args.train_gamma))
        print('Train Randinit {}'.format(args.train_randinit))
        print('SWA={0}, start point={1}, swa_c={2}'.format(args.swa, args.swa_start, args.swa_c_epochs))
        print('LWF={0}, coef_ce={1}, coef_kd1={2}, coef_kd2={3}, start={4}, end={5}'.format(
            args.lwf, args.coef_ce, args.coef_kd1, args.coef_kd2, args.lwf_start, args.lwf_end
        ))

# prepare dataset and models
def setup_dataset_models(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = ResNet18(num_classes = classes)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = ResNet18(num_classes = classes)
            teacher1.normalize = dataset_normalization
            teacher2 = ResNet18(num_classes = classes)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            teacher1.normalize = dataset_normalization
            teacher2 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = 10)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = vgg16_bn(num_classes = 10)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = vgg16_bn(num_classes = 10)
            teacher1.normalize = dataset_normalization
            teacher2 = vgg16_bn(num_classes = 10)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    else:
        raise ValueError("Unknown Model")   
    
    return train_loader, val_loader, test_loader, model, swa_model, teacher1, teacher2

def setup_dataset_models_standard(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = 10)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")   
    
    return train_loader, val_loader, test_loader, model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

# knowledge distillation loss function
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


# training 
def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(input)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_epoch_adv_dual_teacher(train_loader, model, teacher1, teacher2, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.train()
    teacher1.eval()
    teacher2.eval()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)

        with torch.no_grad():
            target_score1 = teacher1(input_adv)
            target_score2 = teacher2(input_adv)

        loss_KD = loss_fn_kd(output_adv, target_score1, T=args.temperature)
        loss_KD2 = loss_fn_kd(output_adv, target_score2, T=args.temperature)

        loss = criterion(output_adv, target)*args.coef_ce + loss_KD*args.coef_kd1 + loss_KD2*args.coef_kd2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

#testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
