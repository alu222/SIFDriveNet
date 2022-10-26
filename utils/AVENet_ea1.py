from image_convnet import *
from audio_convnet import *
import math
from utils.mydata_xu import *
## Main NN starts here
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
from sync_batchnorm import SynchronizedBatchNorm2d

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
		self.fc3     = nn.Linear(128, 3)
		self.softmax = F.softmax

		#EA
		self.norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)
		self.conv1 = nn.Conv2d(128, 128, 1)
		self.k = 64
		self.linear_0 = nn.Conv1d(128, self.k, 1, bias=False)
		self.linear_1 = nn.Conv1d(self.k, 128, 1, bias=False)
		self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)    
		self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),self.norm_layer(128))  
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, _BatchNorm):
				m.weight.data.fill_(1)
				if m.bias is not None:
					m.bias.data.zero_()




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
		x=aud+img
		idn = x
		x = self.conv1(x)
		b, c, h, w = x.size()
		n = h*w
		x = x.view(b, c, h*w)   # b * c * n 
		attn = self.linear_0(x) # b, k, n
		attn = F.softmax(attn, dim=-1) # b, k, n
		attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
		x = self.linear_1(attn) # b, c, n
		x = x.view(b, c, h, w)
		x = self.conv2(x)
		x = x + idn
		x = F.relu(x)
		x=x.squeeze(-1).squeeze(-1)
		#print('x',x.shape)
		out = self.fc3(x)
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
	