'''
Adversarial Training 

'''
import os
import sys 
import torch
import pickle
import argparse
import torch.optim
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.models as models
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Standard Training')

########################## data setting ##########################
parser.add_argument('--data', type=str, default='data/cifar10', help='location of the data corpus', required=True)
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset [cifar10, cifar100, tinyimagenet]', required=True)

########################## model setting ##########################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture [resnet18, wideresnet, vgg16]', required=True)
parser.add_argument('--depth_factor', default=34, type=int, help='depth-factor of wideresnet')
parser.add_argument('--width_factor', default=10, type=int, help='width-factor of wideresnet')

########################## basic setting ##########################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
parser.add_argument('--eval', action="store_true", help="evaluation pretrained model")
parser.add_argument('--print_freq', default=50, type=int, help='logging frequency during training')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

########################## training setting ##########################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')

def main():

    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        print('set random seed = ', args.seed)
        setup_seed(args.seed)

    train_loader, val_loader, test_loader, model = setup_dataset_models_standard(args)
    model.cuda()

    ########################## optimizer and scheduler ##########################
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    ######################### only evaluation ###################################
    if args.eval:
        assert args.pretrained
        pretrained_model = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in pretrained_model.keys():
            pretrained_model = pretrained_model['state_dict']
        model.load_state_dict(pretrained_model)
        test(test_loader, model, criterion, args)
        return 

    os.makedirs(args.save_dir, exist_ok=True)

    ########################## resume ##########################
    start_epoch = 0
    if args.resume:
        print('resume from checkpoint.pth.tar')
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        all_result = checkpoint['result']

    else:
        all_result = {}
        all_result['train_acc'] = []
        all_result['val_sa'] = []
        all_result['test_sa'] = []
        best_sa = 0

    ########################## training process ##########################
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        all_result['train_acc'].append(train_acc)
        scheduler.step()

        ###validation###
        val_sa = test(val_loader, model, criterion, args)
        test_sa = test(test_loader, model, criterion, args)

        all_result['val_sa'].append(val_sa)
        all_result['test_sa'].append(test_sa)

        is_sa_best = val_sa  > best_sa
        best_sa = max(val_sa, best_sa)

        checkpoint_state = {
            'best_sa': best_sa,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'result': all_result
        }

        save_checkpoint(checkpoint_state, is_sa_best, False, False, False, args.save_dir)

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['test_sa'], label='test_SA')
        plt.plot(all_result['val_sa'], label='val_SA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()


if __name__ == '__main__':
    main()


