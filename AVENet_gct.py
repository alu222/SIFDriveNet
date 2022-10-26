from image_convnet import *
from audio_convnet import *

from utils.mydata_xu import *


class GCT(nn.Module):
	def __init__(self, num_channels=128, epsilon=1e-5, mode='l2', after_relu=False):
		super(GCT, self).__init__()
		self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
		self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
		self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
		self.epsilon = epsilon
		self.mode = mode
		self.after_relu = after_relu
	
	def forward(self, x):
		if self.mode == 'l2':
			embedding = (x.pow(2).sum((2, 3), keepdim=True) +self.epsilon).pow(0.5) * self.alpha #[B,C,1,1]
			norm = self.gamma(embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
			# [B,1,1,1],公式中的根号C在mean中体现
		elif self.mode == 'l1':
			if not self.after_relu:
				_x = torch.abs(x)
			else:
				_x = x
			embedding = _x.sum((2, 3), keepdim=True) * self.alpha
			norm = self.gamma(torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
		else:
			print('Unknown mode!')
			sys.exit()
		gate = 1. + torch.tanh(embedding * norm + self.beta)
		# 这里的1+tanh就相当于乘加操作
		return x * gate
		


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

		self.gct=GCT(num_channels=128)


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
		he=img+aud
		he=self.gct(he)
		#mse = self.mse(img, aud, reduce=False).mean(1).unsqueeze(1)
		out = self.fc3(he)
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
	