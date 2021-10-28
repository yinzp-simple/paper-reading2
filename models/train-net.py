from architecture.resnet import ResNet34
from architecture.LeNet import LeNet5, LeNet5Half

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.imagenet import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import collections
import random

class NetTrainer:
    def __init__(self, path_ckpt, path_loss, path_dataset, epochs=10, name_dataset='cifar10', 
				 bs=512, num_epochsaving=100, resume_train=False, path_resume=None):
        # 训练相关参数
        self.last_epoch = 0
        self.epochs = epochs
        self.best_accr = 0
        self.list_loss = []
        self.path_ckpt = path_ckpt
        self.path_loss = path_loss
        
        if name_dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.dataset_train = CIFAR10(path_dataset, transform=transform_train)
            self.dataset_test = CIFAR10(path_dataset, train=False, transform=transform_test)
            self.dataset_train_loader = DataLoader(self.dataset_train, batch_size=bs, shuffle=True, num_workers=8)
            self.dataset_test_loader = DataLoader(self.dataset_test, batch_size=100, num_workers=0)
            
            self.net = ResNet34().cuda()
            self.net = nn.DataParallel(self.net,device_ids=[0,1,2,3], output_device=0)
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
            
            if resume_train:
                ckpt = torch.load(path_resume)
                self.net.load_state_dict(ckpt['net'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                self.last_epoch = ckpt['epoch']
                
        if name_dataset == 'mnist':	
            self.dataset_train = MNIST(path_dataset,download=False, 
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
            self.dataset_test = MNIST(path_dataset,
                            train=False,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))

            self.dataset_train_loader = DataLoader(self.dataset_train, batch_size=256, shuffle=True, num_workers=8)
            self.dataset_test_loader = DataLoader(self.dataset_test, batch_size=1024, num_workers=8)

            self.net = LeNet5Half().cuda()
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [10, 20], gamma=0.1)
            
		
        
    def train(self, save=True):
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.net.train()
            loss_epoch = 0
            for i, (batch_img, batch_label) in enumerate(self.dataset_train_loader, start=1):
                batch_img, batch_label = Variable(batch_img).cuda(non_blocking=True), Variable(batch_label).cuda(non_blocking=True)
                self.optimizer.zero_grad()
                output = self.net(batch_img)
                loss = self.criterion(output, batch_label)
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.data.item()
            # 一个epoch结束
            self.lr_scheduler.step()
            #self.adjust_lr(epoch)
            self.list_loss.append(loss_epoch)
            print('Train-Epoch:%d, Loss:%f, Lr:%f'%(epoch, loss_epoch, self.lr_scheduler.get_last_lr()[0]))
            # 测试
            self.test(epoch)
        if save:
            print("Saving ckpt and loss data")
            # 保存权重
            filename = self.path_ckpt + 'LeNet5Half_ac%f_epoch%d.pth'%(self.best_accr, self.best_state['epoch'])
            torch.save(self.best_state, filename)
            # 保存损失
            lossfile = np.array(self.list_loss)
            np.save(self.path_loss + '/LeNet5Half_loss_{}'.format(self.epochs), lossfile)
        print("Finish Training!Good Luck!:)")

    def test(self, epoch):
        self.net.eval()
        total_correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.dataset_test_loader, start=1):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = self.net(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
            
        acc = float(total_correct) / len(self.dataset_test)
        print('Test-Accuracy:%f' % (acc))
        
        if acc > self.best_accr:
            self.best_accr = acc
            self.best_state = {
			            'net': self.net.state_dict(), 
			            'optimizer':self.optimizer.state_dict(), 
			            'lr_scheduler':self.lr_scheduler.state_dict(),
			            'epoch':epoch
                       }
		
			

  
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    path_current = os.getcwd()
    # path_ckpt = os.path.join(path_current, 'paper-reading/DeepInversion/cache/models/teacher/')
    # path_loss = os.path.join(path_current,'paper-reading/DeepInversion/cache/experimental_data/')
    path_dataset = "/home/ubuntu/datasets/"
    path_ckpt = "/home/ubuntu/YZP/gitee/models/ckpt/"
    path_loss = path_ckpt
    train_teacher = NetTrainer(path_ckpt, path_loss, path_dataset, 
                                   epochs=50, name_dataset='mnist', bs=1024)
    train_teacher.train()