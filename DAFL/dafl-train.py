import sys
sys.path.append("/home/ubuntu/YZP/gitee/models/architecture")
from resnet import ResNet34, ResNet18
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import numpy as np

class Generator(nn.Module):
    """
    Input:[batch, 100]
    Out:[batch, 1, 32, 32]
    """
    def __init__(self, latent=1000):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img




class StudentTrainer:
    def __init__(self, bs=512, epochs=1, lr_G=0.02, lr_S=0.1, 
                 name_dataset='cifar10', path_trained_ckpt=None, path_dataset=None, path_ckpt_saving=None,
                 path_loss_saving=None, ngpus=True) -> None:
        self.generator = Generator().cuda()
        self.teacher = ResNet34().cuda()
        self.teacher.load_state_dict(torch.load(path_trained_ckpt)['net'])
        
        if name_dataset == "cifar10":
            self.student = ResNet18().cuda()
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.dataset_test = CIFAR10(path_dataset, train=False, transform=transform_test)
            self.dataset_test_loader = DataLoader(self.dataset_test, batch_size=bs, num_workers=4)
        if ngpus:
            self.generator = nn.DataParallel(self.generator)
            self.teacher = nn.DataParallel(self.teacher)
            self.student = nn.DataParallel(self.student)
            
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        self.optimizer_S = torch.optim.SGD(self.student.parameters(), lr=lr_S, momentum=0.9, weight_decay=5e-4)
        self.lr_scheduler_S = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_S, [800, 1600])
        # 参数
        self.epochs = epochs
        self.last_epoch = 0
        self.best_accr = 0
        self.list_loss = list()
        self.path_ckpt_saving = path_ckpt_saving
        self.path_loss_saving = path_loss_saving
    def train(self, bs=512, latent=1000, oh=0.05, ie=5, a=0.01):
        self.teacher.eval()
        loss_epoch = 0
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.student.train()
            self.generator.train()
            for i in range(1, 121):
                z = Variable(torch.randn(bs, latent)).cuda()
                self.optimizer_G.zero_grad()
                self.optimizer_S.zero_grad()
                gen_img = self.generator(z)
                output, features = self.teacher(gen_img, out_feature = True)
                pseudo_labels = output.data.max(1)[1]
                # 用于纠正Generator的损失
                ## 损失1：激活损失
                loss_active = -features.abs().mean()
                ## 损失2：分类损失
                loss_onehot = self.criterion(output, pseudo_labels)
                ## 损失3：信息熵损失
                softmax_o_T = F.softmax(output, dim = 1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()

                # 用于纠正学生网络的损失：知识蒸馏损失
                loss_kd = self.kd_loss(self.student(gen_img.detach()), output.detach())
                # 硬损失
                # loss_kd_h = self.criterion(self.student(gen_img.detach()), pseudo_labels)

                loss = oh * loss_onehot + ie*loss_information_entropy + a*loss_active + loss_kd
                loss.backward()
                self.optimizer_S.step()
                self.optimizer_G.step()
                loss_epoch += loss.item()
            # 结束一个epoch
            print('Train-Epoch:{},loss:{},loss_a:{},loss_o:{},loss_ie:{},loss_kd:{}'.format(
                epoch, loss.item(), loss_active, loss_onehot, loss_information_entropy, loss_kd
            ))
            self.list_loss.append(loss_epoch)
            self.test(epoch)
        #结束训练
        print("Finish training")
        torch.save(self.best_state_S, 
                   self.path_loss_saving + 
                   'Resnet18_ac{}epoch{}'.format(self.best_accr,self.best_state_S['epoch']))
        torch.save(self.best_state_G, 
                   self.path_loss_saving + 
                   'generator_{}'.format(self.best_state_G['epoch']))
        
        np.save(self.path_loss_saving+'loss_student_epoch_{}'.format(self.epochs), np.array(self.list_loss))
        print("Finish saving:)")

    def test(self, epoch):
        self.student.eval()
        total_correct = 0
        with torch.no_grad():
            for i ,(images, labels) in enumerate(self.dataset_test_loader):
                images, labels = images.cuda(), labels.cuda()
                output = self.student(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
            accr = float(total_correct) / len(self.dataset_test)
        print('Test--Accuracy:%f'%(accr))
        if accr > self.best_accr:
            self.best_accr = accr
            self.best_state_S = {
                'net':self.student.state_dict(),
                'optimizer':self.optimizer_S.state_dict(),
                'lr_schduler':self.lr_scheduler_S.state_dict(),
                'epoch':epoch
            }
            self.best_state_G = {
                'net':self.generator.state_dict(),
                'optimizer':self.optimizer_G.state_dict(),
                'epoch':epoch
            }
                
                
    def kd_loss(self, y, teacher_scores):
        p = F.log_softmax(y, dim=1)
        q = F.softmax(teacher_scores, dim=1)
        l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
        return l_kl

if __name__ == "__main__":
    path_dataset = '/home/ubuntu/datasets/'
    path_ckpt_saving = '/home/ubuntu/YZP/gitee/models/ckpt/'
    path_loss_saving = '/home/ubuntu/YZP/gitee/paper-reading/DAFL/cache/experimental_data/'
    path_trained_ckpt = '/home/ubuntu/YZP/gitee/models/ckpt/Resnet34_accr0.955800_epoch300.pth'
    train = StudentTrainer(epochs=1, path_dataset=path_dataset, 
                           path_trained_ckpt=path_trained_ckpt, 
                           path_ckpt_saving=path_ckpt_saving,
                           path_loss_saving=path_loss_saving)
    train.train()