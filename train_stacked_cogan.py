import sys
import itertools
import logging
import torch
import torchvision
import torch.nn as nn
from datasets import *
# from dataset_usps import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from trainer_cogan_mnist2usps import *
# from net_config import *
from optparse import OptionParser
import random

from stack_model import *

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"




SPLIT = .75
BATCH_SIZE = 32
LATENT_DIMS = 50
MSE_WEIGHT = 0.01
CLS_WEIGHT = 10.0

TAN_DIR = "./tan_imagery/"
MEX_DIR = "./mex_imagery/"

TAN_TRAIN_FILES = [_ for _ in os.listdir(TAN_DIR) if _.endswith(".png")]
MEX_TRAIN_FILES = [_ for _ in os.listdir(MEX_DIR) if _.endswith(".png")]

TAN_TRAIN_INDICES = random.sample(range(len(TAN_TRAIN_FILES)), int(len(TAN_TRAIN_FILES) * SPLIT))
MEX_TRAIN_INDICES = random.sample(range(len(MEX_TRAIN_FILES)), int(len(MEX_TRAIN_FILES) * SPLIT))
TAN_VAL_INDICES = [_ for _ in range(len(TAN_TRAIN_FILES)) if _ not in TAN_TRAIN_INDICES]
MEX_VAL_INDICES = [_ for _ in range(len(MEX_TRAIN_FILES)) if _ not in MEX_TRAIN_INDICES]

trainer = Tan2MexCoGANTrainer(BATCH_SIZE, LATENT_DIMS)

train_dataset_a = TAN_DATASET(TAN_DIR, TAN_TRAIN_INDICES, "./data/data_for_gan.csv", BATCH_SIZE)

train_dataset_b = MEX_DATASET(MEX_DIR, MEX_TRAIN_INDICES, "./data/data_for_gan.csv", BATCH_SIZE)
test_dataset_b = MEX_DATASET(MEX_DIR, MEX_VAL_INDICES, "./data/data_for_gan.csv", BATCH_SIZE)

print("Number of Tanzania training images: ", len(train_dataset_a))
print("Number of Mexico training images: ", len(train_dataset_b))
print("Number of Mexico validation images: ", len(test_dataset_b))


train_loader_a = torch.utils.data.DataLoader(dataset = train_dataset_a, batch_size = BATCH_SIZE, shuffle = True)
train_loader_b = torch.utils.data.DataLoader(dataset = train_dataset_b, batch_size = BATCH_SIZE, shuffle = True)
test_loader_b = torch.utils.data.DataLoader(dataset = test_dataset_b, batch_size = BATCH_SIZE, shuffle = True)


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def imshow(inp, epoch, it, title = None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))# / 255
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,10))
    plt.imshow(inp)
    plt.savefig(f"./outputs/epoch{str(epoch)}_it{str(it)}.png")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.ioff()
    plt.clf()
    
# We'll use the to keep track of our training stastics (i.e. running training loss, running validation loss, etc...)
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        
        
train_discrim_acc_tracker, train_loss_tracker, train_class_acc_tracker = AverageMeter(), AverageMeter(), AverageMeter()

for epoch in range(0, 10):

    for it, ((images_a, labels_a), (images_b, labels_b)) in enumerate(zip(train_loader_a, train_loader_b)):

        if (images_a.shape[0] == BATCH_SIZE) and (images_b.shape[0] == BATCH_SIZE):

            images_a, labels_a = images_a.cuda(), labels_a.squeeze().cuda()
            images_b = images_b.cuda()
            noise = Variable(torch.randn(BATCH_SIZE, LATENT_DIMS)).cuda()

            ad_acc, mse_loss, cls_acc = trainer.dis_update(images_a, images_b, labels_a, noise)

            noise = Variable(torch.randn(BATCH_SIZE, LATENT_DIMS)).cuda()

            fake_images_a, fake_images_b = trainer.gen_update(noise)

            train_discrim_acc_tracker.update(ad_acc.item())
            train_loss_tracker.update(mse_loss.item())
            train_class_acc_tracker.update(cls_acc.item())

            if it % 10 == 0:

                out = torchvision.utils.make_grid(torch.cat((images_a[0:16].cpu(), fake_images_a[0:16].cpu(), images_b[0:16].cpu(), fake_images_b[0:16].cpu())))
                imshow(out, epoch, it, title = it) 
        
        
    
    print("Epoch: ", epoch)     
    print("Average Discriminator Accuracy: ", train_discrim_acc_tracker.avg)
    print("Average Loss: ", train_loss_tracker.avg)
    print("Average Class Accuracy: ", train_class_acc_tracker.avg, "\n")
    
    
    with open("./log.txt", "a") as f:
        f.write("Epoch: " + str(epoch) + "\n" + \
                "Average Discriminator Accuracy: " + str(train_discrim_acc_tracker.avg) + "\n" + \
                "Average Loss: " + str(train_loss_tracker.avg) + "\n" + \
                "Average Class Accuracy: " + str(train_class_acc_tracker.avg) + "\n\n")