from image_convnet import *
from audio_convnet import *
from AVENet import *
from utils.dataloader import *

import argparse
from torch.optim import *
from torchvision.transforms import *
import warnings
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils.mydata import *

import json

warnings_file = open("warning_logs.txt", "w+")
def customwarn(message, category, filename, lineno, file=None, line=None):
    warnings_file.write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = customwarn


choices = ["demo", "main", "checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio", \
			"imageToImageQueries", "crossModalQueries"]


# Write parser
parser = argparse.ArgumentParser(description="Select code to run.")
parser.add_argument('--mode', required=True, choices=choices, type=str,default="main")

# Demo to check if things are working
def demo():
	model = AVENet()
	image = Variable(torch.rand(2, 3, 224, 224))
	audio = Variable(torch.rand(2, 1, 257, 200))

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)


# data_transforms = {
#         'train': Compose([
#             Resize(size=(224, 224)),

#             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]),
#         'val': Compose([
#             Resize(size=(224, 224)),
#             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }

# Main function here
def main(use_cuda=True, lr=0.25e-4, EPOCHS=10, save_checkpoint=500, batch_size=2, model_name="avenet.pt"):
	
	lossfile = open("losses.txt", "a+")
	print("Using batch size: %d"%batch_size)
	print("Using lr: %f"%lr)

	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	train_dir='./train.txt'
	train_datasets = Mydata(train_dir)
	train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, num_workers=0)
	#dataset = GetAudioVideoDataset()
	#valdataset = GetAudioVideoDataset(video_path="Video_val/", audio_path="Audio_val/", validation=True)
	#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	#valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	# optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
	optim = SGD(model.parameters(), lr=lr)
	print("Optimizer loaded.")
	model.train()

	try:
		for epoch in range(EPOCHS):
			# Run algo
			for subepoch, (img, out, aud) in enumerate(train_loader):
				optim.zero_grad()

				# Filter the bad ones first
				'''out = out.squeeze(1)#第1维的维度值为1，则去掉该维度，否则tensor不变  [batch]
				idx = (out != 2).numpy().astype(bool)#转换为bool型的  [T T F T]
				if idx.sum() == 0:  idx.sum() 3  选出标签不是2的数量，说明有正确的257*200频谱图
					continue  全都是不正确的257*200频谱图

				# Find the new variables
				img = torch.Tensor(img.numpy()[idx, :, :, :])
				aud = torch.Tensor(aud.numpy()[idx, :, :, :])
				out = torch.LongTensor(out.numpy()[idx])

				# Print shapes
				img = Variable(img)
				aud = Variable(aud)
				out = Variable(out)

				# print(img.shape, aud.shape, out.shape)'''

				M = img.shape[0] 
				

				if use_cuda:
					img = img.cuda()
					aud = aud.cuda()
					out = out.cuda()
				img = Variable(img)
				aud = Variable(aud)
				out = Variable(out)

				o, _, _ = model(img, aud)
				# print(o)
				# print(o.shape, out.shape)
				loss = crossEntropy(o, out)
				loss.backward()
				optim.step()

				# Calculate accuracy
				_, ind = o.max(1) #每行最大值
				accuracy = (ind.data == out.data).sum()*1.0/M
				'''
				# Periodically print subepoch values
				if subepoch%10 == 0:
					model.eval()
					for (img, aud, out) in valdataloader:
						break
					# Filter the bad ones first
					out = out.squeeze(1)
					idx = (out != 2).numpy().astype(bool)
					if idx.sum() == 0:
						continue
					# Find the new variables
					img = torch.Tensor(img.numpy()[idx, :, :, :])
					aud = torch.Tensor(aud.numpy()[idx, :, :, :])
					out = torch.LongTensor(out.numpy()[idx])

					# Print shapes
					img = Variable(img, volatile=True)
					aud = Variable(aud, volatile=True)
					out = Variable(out, volatile=True)

					# print(img.shape, aud.shape, out.shape)

					M = img.shape[0]
					if use_cuda:
						img = img.cuda()
						aud = aud.cuda()
						out = out.cuda()

					o, _, _ = model(img, aud)
					valloss = crossEntropy(o, out)
					# Calculate valaccuracy
					_, ind = o.max(1)
					valaccuracy = (ind.data == out.data).sum()*1.0/M'''

				print("Epoch: %d, Subepoch: %d, Loss: %f,acc: %f"%(epoch, subepoch, loss.item(),accuracy))
					#lossfile.write("Epoch: %d, Subepoch: %d, Loss: %f, Valloss: %f, batch_size: %d, acc: %f, valacc: %f\n"%(epoch, subepoch, loss.data[0], valloss.data[0], M, accuracy, valaccuracy))
					#model.train()
				
				# Save model'''
				'''if subepoch%save_checkpoint == 0 and subepoch > 0:
					torch.save(model.state_dict(), model_name)
					print("Checkpoint saved.")'''

	except Exception as e:
		print(e)
		torch.save(model.state_dict(), "backup"+model_name)
		print("Checkpoint saved and backup.")

	#lossfile.close()



def getAVENet(use_cuda):
	model = AVENet()
	# model.fc3.weight.data[0] = -0.1
	# model.fc3.weight.data[1] =  0.1
	# model.fc3.bias.data[0] =   1.0
	# model.fc3.bias.data[1] = - 1.0
	model.fc3.weight.data[0] = -0.7090
	model.fc3.weight.data[1] =  0.7090
	model.fc3.bias.data[0] =   1.2186
	model.fc3.bias.data[1] = - 1.2186
	if use_cuda:
		model = model.cuda()

	return model


def checkValidation(use_cuda=True, batch_size=64, model_name="avenet.pt", validation=True):
	
	print("Using batch size: %d"%batch_size)
	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")


	print("Model name: {0}".format(model_name))
	if validation == True or validation == "validation":
		print("Using validation")
		dataset = GetAudioVideoDataset(video_path="Video_val/", audio_path="Audio_val/", validation=True)
	elif validation == "test":
		print("Using test.")
		dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test")
	else:
		print("Using training")
		dataset = GetAudioVideoDataset()


	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")
	model.eval()
	correct = []
	count = []
	losses= []
	print(len(dataset))

	try:
		for subepoch, (img, aud, out) in enumerate(dataloader):
			# Filter the bad ones first
			out = out.squeeze(1)
			idx = (out != 2).numpy().astype(bool)
			if idx.sum() == 0:
				continue

			# Find the new variables
			img = torch.Tensor(img.numpy()[idx, :, :, :])
			aud = torch.Tensor(aud.numpy()[idx, :, :, :])
			out = torch.LongTensor(out.numpy()[idx])

			# Print shapes
			img = Variable(img, volatile=True)
			aud = Variable(aud, volatile=True)
			out = Variable(out, volatile=True)

			# print(img.shape, aud.shape, out.shape)

			M = img.shape[0]
			if use_cuda:
				img = img.cuda()
				aud = aud.cuda()
				out = out.cuda()

			o, _, _ = model(img, aud)

			loss = crossEntropy(o, out).data[0]
			# Calculate accuracy
			_, ind = o.max(1)
			acc = (ind.data == out.data).sum()*1.0
			correct.append(acc)
			losses.append(loss)
			count.append(M)
			print(subepoch, acc, M, acc/M, loss)
			if subepoch == 100:
				break

	except Exception as e:
		print(e)

	M = sum(count)
	corr = sum(correct)
	print("Total frames: {0}:".format(M))
	print("Total correct: {0}".format(corr))
	print("Total accuracy: {0}".format(corr/M))
	print("Total loss: {0}".format(np.mean(losses)))



def reverseTransform(img, aud):
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]

	for i in range(3):
		img[:,i,:,:] = img[:,i,:,:]*std[i] + mean[i]

	return img, aud



def bgr2rgb(img):
	res = img+0.0
	if len(res.shape) == 4:
		res[:,0,:,:] = img[:,2,:,:]
		res[:,2,:,:] = img[:,0,:,:]
	else:
		res[0,:,:] = img[2,:,:]
		res[2,:,:] = img[0,:,:]
	return res






if __name__ == "__main__":
	cuda = True
	args = parser.parse_args()
	mode = args.mode

	if mode == "demo":
		demo()

	elif mode == "main":
		main(use_cuda=cuda, batch_size=64)

	elif mode == "checkValidation":
		checkValidation(use_cuda=cuda, validation="test", batch_size=128, model_name="models/avenet.pt")

	warnings_file.close()
