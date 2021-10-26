import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.imagenet import ImageNet
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import collections
import random
import resnet



class InputTrainer:
    def __init__(self, path_ckpt_teacher, path_inputs_saving, bs=8):
        self.bs = bs
        # teacher and student net
        ckpt = torch.load(path_ckpt_teacher)
        print('Using trained teacher network...')
        self.teacher = resnet.ResNet34().cuda()
        self.teacher.load_state_dict(ckpt['net'])
        #self.teacher = nn.DataParallel(self.teacher, device_ids=[0,1,2])
        self.student = resnet.ResNet18().cuda()
        self.teacher.eval()
        self.student.eval()
        # 给BN添加hook函数
        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.register_forward_hook(self.hook_fn)
        
        self.inputs = torch.randn((self.bs, 3, 32, 32), requires_grad=True, device='cuda', dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer_in = torch.optim.Adam([self.inputs], lr=0.1)
        self.optimizer_in.state = collections.defaultdict(dict)
        #arguments
        self.last_epoch = 0
        self.loss_r_feature = list()
        self.path_inputs_saving = path_inputs_saving
        
    def train(self, epochs, l2_coeff=0.0, var_scale=0.001, bn_reg_scale=10):
        # batch size 250+6
        targets = torch.LongTensor([0,1,2,3,4,5,6,7,8,9]*3 + [0,1]).cuda()
        largest_loss = 1e6
        for epoch in range(self.last_epoch+1, epochs+1):
            off1 = random.randint(-2,-2)
            off2 = random.randint(-2,-2)
            # shift the input along the second(w) and third(h) axis by a random number
            inputs_jit = torch.roll(self.inputs, shifts=(off1, off2), dims=(2,3))
            
            self.optimizer_in.zero_grad()
            self.teacher.zero_grad()
            outputs = self.teacher(inputs_jit)
            ## 损失1：分类错误损失
            loss = self.criterion(outputs, targets)
            ## 损失2.1：l2 norm损失
            loss = loss+l2_coeff*torch.norm(inputs_jit, 2)
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            ## 损失2.2：总方差损失
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss+var_scale*loss_var
            # 损失3：特征分布正则化损失
            loss_distr = sum([r_feature for r_feature in self.loss_r_feature])
            # loss = loss+bn_reg_scale*loss_distr
            #loss = loss_c + l2_coeff*loss_l2 + var_scale*loss_var + bn_reg_scale*loss_distr
            ## 梯度下降
            loss.backward()
            self.optimizer_in.step()
            # 采用这种保存方法，节省保存次数
            if largest_loss > loss.item():
                largest_loss = loss.item()
                best_inputs = self.inputs.data
            
            print('train:epoch:%d,loss_tv:%f,loss_f:%f,total:%f'
                  %(epoch, loss_var,loss_distr, loss.data.item()))
            
        vutils.save_image(best_inputs[:20].clone(), self.path_inputs_saving, 
                          normalize=True, scale_each=True, nrow=10)
            
    def hook_fn(self, module, input_data, output_data):
        num_ch = input_data[0].shape[1]
        mean = input_data[0].mean([0,2,3])
        #var = input_data[0].permute(1,0,2,3).contiguous().view([num_ch, -1]).var(1, unbiased=False)
        var = input_data[0].var([0, 2, 3], unbiased=False)
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(module.running_mean.data.type(mean.type()) - mean, 2)
        self.loss_r_feature.append(r_feature)
        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,3'
    path_current = os.getcwd()

    path_ckpt_teacher = os.path.join(path_current, 'cache/models/teacher/teacher')
    path_save_img = os.path.join(path_current, 'cache/best_img2k.png')
    # ckpt = torch.load(path_ckpt_teacher)
    # net = ResNet34().cuda()
    # net.load_state_dict(ckpt['net'])
    trainer = InputTrainer(path_ckpt_teacher, path_save_img, bs=32)
    trainer.train(2000)