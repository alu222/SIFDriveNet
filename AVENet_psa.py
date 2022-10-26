from image_convnet import *
from audio_convnet import *

from utils.mydata_xu import *

import math
import torch
import torch.nn as nn
from .SE_weight_module import SEWeightModule

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out
## Main NN starts here
class AVENet(nn.Module):

	def __init__(self):
		super(AVENet, self).__init__()

		self.relu   = F.relu
		self.imgnet = ImageConvNet()
		self.audnet = AudioConvNet()

		# Vision subnetwork
		self.vpool4  = nn.MaxPool2d(14, stride=14)
		self.vfc1    = nn.Linear(512, 128)
		self.vfc2    = nn.Linear(128, 128)
		self.vl2norm = nn.BatchNorm1d(128)

		# Audio subnetwork
		self.apool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.afc1    = nn.Linear(512, 128)
		self.afc2    = nn.Linear(128, 128)
		self.al2norm = nn.BatchNorm1d(128)

		# Combining layers
		self.mse     = F.mse_loss
		#self.fc3     = nn.Linear(1, 2)
		self.fc3     = nn.Linear(1, 3)
		self.softmax = F.softmax

		#psa
		self.psa = PSAModule(128, 64, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16])


	def forward(self, image, audio):
		# Image
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)

		# Audio
		aud = self.audnet(audio)
		aud = self.apool4(aud).squeeze(2).squeeze(2)
		aud = self.relu(self.afc1(aud))
		aud = self.afc2(aud)
		aud = self.al2norm(aud)

		# Join them 
		#mse = self.mse(img, aud, reduce=False).mean(1).unsqueeze(1)
		aud=aud.unsqueeze(-1).unsqueeze(-1)
		img=img.unsqueeze(-1).unsqueeze(-1)
		zong=aud+img
		zong=self.psa(zong)
		out = self.fc3(zong)
		#out = self.softmax(out, 1)#对每一行进行softmax

		return out, img, aud


	def get_image_embeddings(self, image):
		# Just get the image embeddings
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)
		return img

if __name__ == '__main__':
	model = AVENet().cuda()
	
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	speed = Variable(torch.rand(2, 1, 257, 200)).cuda()

	# Run a feedforward and check shape
	o,_,_ = model(image,speed)

	print(o.shape)#[2,3]
	