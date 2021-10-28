import sys
sys.path.append("/home/ubuntu/YZP/gitee/models/architecture/")
from LeNet import LeNet5, LeNet5Half
from gan import GeneratorA
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


class DaflTrainer:
    def __init__(self, epochs=10, bs=256, path_ckpt_teacher=None, path_ckpt_student=None, path_ckpt_gan=None, 
                 path_loss=None, path_dataset=None, 
                 lr_S=0.01, lr_G=1e-3) -> None:
        # 1. 训练参数
        self.epochs = epochs
        self.last_epoch = 0
        self.bs = bs
        self.loss_list_S = list()
        self.loss_list_G = list()
        self.best_accr = 0
        self.path_ckpt_student = path_ckpt_student
        self.path_loss = path_loss
        # 2. 模型载入
        self.teacher = LeNet5().cuda()
        self.student = LeNet5Half().cuda()
        self.generator = GeneratorA(nz=100, nc=1, img_size=32).cuda()
        self.teacher.load_state_dict(torch.load(path_ckpt_teacher)['net'])
        self.teacher.eval()
        # 3. 数据集准备
        self.dataset_test = MNIST(path_dataset,
                        train=False,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))

        self.dataset_test_loader = DataLoader(self.dataset_test, batch_size=1024, num_workers=8)
        # 4. 优化器
        self.optimizer_S = torch.optim.SGD(self.student.parameters(), lr=lr_S, weight_decay=1e-4, momentum=0.9)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        self.lr_scheduler_S = torch.optim.lr_scheduler.StepLR(self.optimizer_S, 100, gamma=0.1)
        self.lr_scheduler_S = torch.optim.lr_scheduler.StepLR(self.optimizer_G, 100, gamma=0.1)  
    def train(self, epoch_itrs=50, flag_save=None):
        for epoch in range(self.last_epoch+1, self.epochs+1):
            loss_G_epoch = 0
            loss_S_epoch = 0
            self.student.train()
            for it in range(epoch_itrs):
                
                for k in range(5):
                    z = torch.randn((self.bs, 100, 1, 1)).cuda()
                    self.optimizer_S.zero_grad()
                    fake = self.generator(z).detach()
                    pseudo_labels = self.teacher(fake)
                    outputs = self.student(fake)
                    loss_S = F.l1_loss(outputs, pseudo_labels.detach())
                    
                    loss_S.backward()
                    self.optimizer_S.step()
                    loss_S_epoch += loss_S.item()
                z = torch.randn((self.bs, 100, 1, 1)).cuda()
                self.optimizer_G.zero_grad()
                self.generator.train()
                fake = self.generator(z)
                pseudo_labels = self.teacher(fake)
                outputs = self.student(fake)
                loss_G = -F.l1_loss(outputs, pseudo_labels)
                
                loss_G.backward()
                self.optimizer_G.step()
                loss_G_epoch += loss_G.item()
            # 结束一个epoch
            print('Train Epoch:{}:Loss_S:{:.4f} Loss_G:{:.3f}'.format(
                epoch, loss_S_epoch, loss_G_epoch
            ))
            self.test(epoch)
            self.loss_list_S.append(loss_S_epoch)
            self.loss_list_G.append(loss_G_epoch)
        
        print("Saving ckpt and loss data")
        # 保存权重
        filename = self.path_ckpt_student + 'DFAD_LeNet5Half_ac%f_epoch%d.pth'%(self.best_accr, self.best_state['epoch'])
        torch.save(self.best_state, filename)
        lossfile_S = np.array(self.loss_list_S)
        lossfile_G = np.array(self.loss_list_G)
        np.save(self.path_loss + flag_save+'LeNet5Half_loss_{}'.format(self.epochs), lossfile_S)
        np.save(self.path_loss + flag_save+'GeneratorA_loss_{}'.format(self.epochs), lossfile_G)
       
        
        print("Finish Training!Good Luck!:)")
    
    def test(self, epoch):
        self.student.eval()
        total_correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.dataset_test_loader, start=1):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = self.student(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
            
        acc = float(total_correct) / len(self.dataset_test)
        print('Test-Accuracy:{:.3f}%'.format(acc*100))
        
        if acc > self.best_accr:
            self.best_accr = acc
            self.best_state = {
			            'net': self.student.state_dict(), 
			            'optimizer':self.optimizer_S.state_dict(), 
			            #'lr_scheduler':self.lr_scheduler.state_dict(),
			            'epoch':epoch
                       }
            

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    path_current = os.getcwd()
    # path_ckpt = os.path.join(path_current, 'paper-reading/DeepInversion/cache/models/teacher/')
    # path_loss = os.path.join(path_current,'paper-reading/DeepInversion/cache/experimental_data/')
    path_dataset = "/home/ubuntu/datasets/"
    path_ckpt_save = "/home/ubuntu/YZP/gitee/models/ckpt/"
    path_loss = "/home/ubuntu/YZP/gitee/models/ckpt/lossfile/"
    train = DaflTrainer(epochs=40, bs=512, path_ckpt_teacher="/home/ubuntu/YZP/gitee/models/ckpt/LeNet5_ac0.992400_epoch12.pth", path_ckpt_student=path_ckpt_save, 
                        path_dataset=path_dataset, path_loss=path_loss)
    train.train(flag_save="5")
                
        
        