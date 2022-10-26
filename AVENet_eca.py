from multiprocessing.util import sub_debug
from image_convnet import *
from audio_convnet import *

from utils.mydata_xu import *
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

		t = int(abs((math.log(128, 2) + 1) / 2))
		k = t if t % 2 else t + 1
		self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
		self.sigmoid = nn.Sigmoid()



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
		he=img+aud
		he_1=self.conv1(he.squeeze(-1).transpose(-1,-2))
		he_1=he_1.transpose(-1,-2).unsqueeze(-1)
		he_1=self.sigmoid(he_1)
		he=he*he_1
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
	