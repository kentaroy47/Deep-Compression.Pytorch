# -*- coding: utf-8 -*-

'''Deep Compression with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning')
parser.add_argument('--loadfile', '-l', default="checkpoint/ckpt.t7",dest='loadfile')
parser.add_argument('--prune', '-p', default=0.5, dest='prune', help='Parameters to be pruned')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
args = parser.parse_args()

prune = args.prune

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load weights from checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isfile(args.loadfile), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.loadfile)
net.load_state_dict(checkpoint['net'])
    
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def prune_weights(torchweights):
    weights=np.abs(torchweights.cpu().numpy());
    weightshape=weights.shape
    rankedweights=weights.reshape(weights.size).argsort()#.reshape(weightshape)
    
    num = weights.size
    prune_num = int(np.round(num*prune))
    count=0
    masks = np.zeros_like(rankedweights)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n]=1
        else: count+=1
    print("total weights:", num)
    print("weights pruned:",count)
    
    masks=masks.reshape(weightshape)
    weights=masks*weights
    
    return torch.from_numpy(weights).cuda(), masks
    
# prune weights
addressbook=[]
maskbook=[]
for k, v in net.state_dict().items():
    if "conv2" in k:
        addressbook.append(k)
        print("pruning layer:",k)
        weights=v
        weights, masks = prune_weights(weights)
        maskbook.append(masks)
        checkpoint['net'][k] = weights
        
checkpoint['address'] = addressbook
checkpoint['mask'] = maskbook
net.load_state_dict(checkpoint['net'])

# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # mask pruned weights 
        checkpoint['net']=net.state_dict()
#        print("zeroing..")
#        print(np.count_nonzero(checkpoint['net'][addressbook[0]].cpu().numpy()))
        for address, mask in zip(addressbook, maskbook):
#            print(address)
            checkpoint['net'][address] = torch.from_numpy(checkpoint['net'][address].cpu().numpy() * mask)
#        print(np.count_nonzero(checkpoint['net'][addressbook[0]].cpu().numpy()))  
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/pruned_ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+20):
    train(epoch)
    test(epoch)