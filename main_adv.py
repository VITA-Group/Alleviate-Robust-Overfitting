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

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

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

########################## attack setting ##########################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8, type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2, type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

########################## SWA setting ##########################
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=55, metavar='N', help='SWA start epoch number (default: 55)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')

########################## KD setting ##########################
parser.add_argument('--lwf', action='store_true', help='lwf usage flag (default: off)')
parser.add_argument('--t_weight1', type=str, default=None, required=False, help='pretrained weight for teacher1')
parser.add_argument('--t_weight2', type=str, default=None, required=False, help='pretrained weight for teacher2')
parser.add_argument('--coef_ce', type=float, default=0.3, help='coef for CE')
parser.add_argument('--coef_kd1', type=float, default=0.1, help='coef for KD1')
parser.add_argument('--coef_kd2', type=float, default=0.6, help='coef for KD2')
parser.add_argument('--temperature', type=float, default=2.0, help='temperature of knowledge distillation loss')
parser.add_argument('--lwf_start', type=int, default=0, metavar='N', help='start point of lwf (default: 200)')
parser.add_argument('--lwf_end', type=int, default=200, metavar='N', help='end point of lwf (default: 200)')


def main():

    args = parser.parse_args()
    args.train_eps = args.train_eps / 255
    args.train_gamma = args.train_gamma / 255
    args.test_eps = args.test_eps / 255
    args.test_gamma = args.test_gamma / 255
    print_args(args)
    print(args)


    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        print('set random seed = ', args.seed)
        setup_seed(args.seed)

    train_loader, val_loader, test_loader, model, swa_model, teacher1, teacher2 = setup_dataset_models(args)

    if args.swa:
        swa_model.cuda()
        swa_n = 0        
    if args.lwf:
        teacher1.cuda()
        teacher2.cuda()

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
        if args.swa:
            print('loading from swa_state_dict')
            pretrained_model = pretrained_model['swa_state_dict']
        else:
            print('loading from state_dict')
            if 'state_dict' in pretrained_model.keys():
                pretrained_model = pretrained_model['state_dict']
        model.load_state_dict(pretrained_model)
        test(test_loader, model, criterion, args)
        test_adv(test_loader, model, criterion, args)
        return 

    os.makedirs(args.save_dir, exist_ok=True)

    ########################## loading teacher model weight ##########################
    if args.lwf:
        print('loading teacher model')
        t1_checkpoint = torch.load(args.t_weight1, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in t1_checkpoint.keys():
            t1_checkpoint = t1_checkpoint['state_dict']
        teacher1.load_state_dict(t1_checkpoint)
        t2_checkpoint = torch.load(args.t_weight2, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in t2_checkpoint.keys():
            t2_checkpoint = t2_checkpoint['state_dict']
        teacher2.load_state_dict(t2_checkpoint)

        print('test for teacher1')
        test(test_loader, teacher1, criterion, args)
        test_adv(test_loader, teacher1, criterion, args)
        print('test for teacher2')                
        test(test_loader, teacher2, criterion, args)
        test_adv(test_loader, teacher2, criterion, args)

    ########################## resume ##########################
    start_epoch = 0
    if args.resume:
        print('resume from checkpoint.pth.tar')
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        best_ra = checkpoint['best_ra']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        all_result = checkpoint['result']

        if args.swa:
            best_sa_swa = checkpoint['best_sa_swa']
            best_ra_swa = checkpoint['best_ra_swa']
            swa_model.load_state_dict(checkpoint['swa_state_dict'])
            swa_n = checkpoint['swa_n']
    else:
        all_result = {}
        all_result['train_acc'] = []
        all_result['val_sa'] = []
        all_result['val_ra'] = []
        all_result['test_sa'] = []
        all_result['test_ra'] = []
        best_sa = 0
        best_ra = 0

        if args.swa:
            all_result['val_sa_swa'] = []
            all_result['val_ra_swa'] = []
            all_result['test_sa_swa'] = []
            all_result['test_ra_swa'] = []
            swa_n = 0  
            best_sa_swa = 0
            best_ra_swa = 0

    is_sa_best = False
    is_ra_best = False
    is_sa_best_swa = False
    is_ra_best_swa = False

    ########################## training process ##########################
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])

        if args.lwf and epoch >= args.lwf_start and epoch < args.lwf_end:
            print('adversarial training with LWF')
            train_acc = train_epoch_adv_dual_teacher(train_loader, model, teacher1, teacher2, criterion, optimizer, epoch, args)
        else:
            print('baseline adversarial training')
            train_acc = train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args)

        all_result['train_acc'].append(train_acc)
        scheduler.step()

        ###validation###
        val_sa = test(val_loader, model, criterion, args)
        val_ra = test_adv(val_loader, model, criterion, args)   
        test_sa = test(test_loader, model, criterion, args)
        test_ra = test_adv(test_loader, model, criterion, args)  

        all_result['val_sa'].append(val_sa)
        all_result['val_ra'].append(val_ra)
        all_result['test_sa'].append(test_sa)
        all_result['test_ra'].append(test_ra)

        is_sa_best = val_sa  > best_sa
        best_sa = max(val_sa, best_sa)

        is_ra_best = val_ra  > best_ra
        best_ra = max(val_ra, best_ra)

        checkpoint_state = {
            'best_sa': best_sa,
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'result': all_result
        }

        if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:

            # SWA
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(train_loader, swa_model)

            val_sa_swa = test(val_loader, swa_model, criterion, args)
            val_ra_swa = test_adv(val_loader, swa_model, criterion, args)   
            test_sa_swa = test(test_loader, swa_model, criterion, args)
            test_ra_swa = test_adv(test_loader, swa_model, criterion, args)  

            all_result['val_sa_swa'].append(val_sa_swa)
            all_result['val_ra_swa'].append(val_ra_swa)
            all_result['test_sa_swa'].append(test_sa_swa)
            all_result['test_ra_swa'].append(test_ra_swa)

            is_sa_best_swa = val_sa_swa  > best_sa_swa
            best_sa_swa = max(val_sa_swa, best_sa_swa)

            is_ra_best_swa = val_ra_swa  > best_ra_swa
            best_ra_swa = max(val_ra_swa, best_ra_swa)

            checkpoint_state.update({
                'swa_state_dict': swa_model.state_dict(),
                'swa_n': swa_n,
                'best_sa_swa': best_sa_swa,
                'best_ra_swa': best_ra_swa
            })

        elif args.swa:

            all_result['val_sa_swa'].append(val_sa)
            all_result['val_ra_swa'].append(val_ra)
            all_result['test_sa_swa'].append(test_sa)
            all_result['test_ra_swa'].append(test_ra)

        checkpoint_state.update({
            'result': all_result
        })
        save_checkpoint(checkpoint_state, is_sa_best, is_ra_best, is_sa_best_swa, is_ra_best_swa, args.save_dir)

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['test_sa'], label='SA')
        plt.plot(all_result['test_ra'], label='RA')

        if args.swa:
            plt.plot(all_result['test_sa_swa'], label='SWA_SA')
            plt.plot(all_result['test_ra_swa'], label='SWA_RA')

        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()


if __name__ == '__main__':
    main()


