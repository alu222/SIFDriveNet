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
import json
'/home/lulu/Documents/speed/objects-that-sound/train.txt'
## Handle warnings here
# CITE: https://stackoverflow.com/questions/858916/how-to-redirect-python-warnings-to-a-custom-stream
warnings_file = open("warning_logs.txt", "w+")
def customwarn(message, category, filename, lineno, file=None, line=None):
    warnings_file.write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = customwarn


choices = ["demo", "main", "checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio", \
			"imageToImageQueries", "crossModalQueries"]


# Write parser
parser = argparse.ArgumentParser(description="Select code to run.")
parser.add_argument('--mode', required=True, choices=choices, type=str)

# Demo to check if things are working
def demo():
	model = AVENet()
	image = Variable(torch.rand(2, 3, 224, 224))
	audio = Variable(torch.rand(2, 1, 257, 200))

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)

# Main function here
def main(use_cuda=True, lr=0.25e-4, EPOCHS=100, save_checkpoint=500, batch_size=64, model_name="avenet.pt"):
	
	lossfile = open("losses.txt", "a+")
	print("Using batch size: %d"%batch_size)
	print("Using lr: %f"%lr)

	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")


	dataset = GetAudioVideoDataset()
	valdataset = GetAudioVideoDataset(video_path="Video_val/", audio_path="Audio_val/", validation=True)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	# optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
	optim = SGD(model.parameters(), lr=lr)
	print("Optimizer loaded.")
	model.train()

	try:
		for epoch in range(EPOCHS):
			# Run algo
			for subepoch, (img, aud, out) in enumerate(dataloader):
				optim.zero_grad()

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
				img = Variable(img)
				aud = Variable(aud)
				out = Variable(out)

				# print(img.shape, aud.shape, out.shape)

				M = img.shape[0]
				if use_cuda:
					img = img.cuda()
					aud = aud.cuda()
					out = out.cuda()

				o, _, _ = model(img, aud)
				# print(o)
				# print(o.shape, out.shape)
				loss = crossEntropy(o, out)
				loss.backward()
				optim.step()

				# Calculate accuracy
				_, ind = o.max(1)
				accuracy = (ind.data == out.data).sum()*1.0/M

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
					valaccuracy = (ind.data == out.data).sum()*1.0/M

					print("Epoch: %d, Subepoch: %d, Loss: %f, Valloss: %f, batch_size: %d, acc: %f, valacc: %f"%(epoch, subepoch, loss.data[0], valloss.data[0], M, accuracy, valaccuracy))
					lossfile.write("Epoch: %d, Subepoch: %d, Loss: %f, Valloss: %f, batch_size: %d, acc: %f, valacc: %f\n"%(epoch, subepoch, loss.data[0], valloss.data[0], M, accuracy, valaccuracy))
					model.train()
				
				# Save model
				if subepoch%save_checkpoint == 0 and subepoch > 0:
					torch.save(model.state_dict(), model_name)
					print("Checkpoint saved.")

	except Exception as e:
		print(e)
		torch.save(model.state_dict(), "backup"+model_name)
		print("Checkpoint saved and backup.")

	lossfile.close()



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
	

def getVideoEmbeddings(model_name="avenet.pt"):
	# Get video embeddings on the test set
	dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test", return_tags=True)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	print("Loading data.")
	for img, aud, res, vidTags, audTags in dataloader:
		break

	model = getAVENet(True)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	model.eval()
	embed = model.get_image_embeddings(Variable(img, volatile=True).cuda())
	print(embed.data)

	embed = embed.data.cpu().numpy()
	vidTags = vidTags.numpy()

	data = TSNE(n_iter=5000).fit_transform(embed)
	X = data[:,0]
	Y = data[:,1]
	print(X.shape, Y.shape, vidTags.shape)
	for i in range(X.shape[0]):
		plt.scatter(X[i], Y[i])

	for i in range(X.shape[0]):
		plt.annotate(vidTags[i,0], (X[i], Y[i]))

	print(vidTags)
	plt.show()


def reverseTransform(img, aud):
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]

	for i in range(3):
		img[:,i,:,:] = img[:,i,:,:]*std[i] + mean[i]

	return img, aud


def generateEmbeddingsForVideoAudio(model_name="avenet.pt", use_cuda=True):
	# Get video embeddings on the test set
	dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test",\
	 return_tags=True, return_audio=True)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	print("Loading data.")
	# for img, aud, res, vidTags, audTags, audioSample in dataloader:
	# 	break

	model = getAVENet(True)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	imgList, audList, resList, vidTagList, audTagList = [], [], [], [], []
	imgEmbedList, audEmbedList = [], []
	audioSampleList = []

	model.eval()
	for i, (img, aud, res, vidTag, audTag, audSamples) in enumerate(dataloader):
		# Filter the bad ones first
		res = res.squeeze(1)
		idx = (res != 2).numpy().astype(bool)
		if idx.sum() == 0:
			continue

		# Find the new variables
		img = torch.Tensor(img.numpy()[idx, :, :, :])
		aud = torch.Tensor(aud.numpy()[idx, :, :, :])
		res = torch.LongTensor(res.numpy()[idx])
		vidTag = vidTag.numpy()[idx]
		audTag = audTag.numpy()[idx]
		audSamples = audSamples.numpy()[idx]

		img = Variable(img, volatile=True)
		aud = Variable(aud, volatile=True)
		res = Variable(res, volatile=True)

		M = img.shape[0]
		if use_cuda:
			img = img.cuda()
			aud = aud.cuda()
			res = res.cuda()

		o, imgEmbed, audEmbed = model(img, aud)
		_, ind = o.max(1)

		
		# Grab the correct indices
		idx = ((ind == res) * (res == 0)).data.cpu().numpy().astype(bool)
		print(i)

		img, aud = reverseTransform(img, aud)
		# plt.imshow(img[0].data.cpu().numpy().transpose(1,2,0))
		# plt.draw()
		# plt.pause(0.005)

		imgList.append(img.data.cpu().numpy()[idx, :])
		audList.append(aud.data.cpu().numpy()[idx, :])
		imgEmbedList.append(imgEmbed.data.cpu().numpy()[idx, :])
		audEmbedList.append(audEmbed.data.cpu().numpy()[idx, :])
		vidTagList.append(vidTag[idx])
		audTagList.append(audTag[idx])
		audioSampleList.append(audSamples[idx])

		if i == 35:
			break

	torch.save([imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList, audioSampleList], "savedEmbeddings.pt")


def bgr2rgb(img):
	res = img+0.0
	if len(res.shape) == 4:
		res[:,0,:,:] = img[:,2,:,:]
		res[:,2,:,:] = img[:,0,:,:]
	else:
		res[0,:,:] = img[2,:,:]
		res[2,:,:] = img[0,:,:]
	return res


def getNumToTagsMap():
	with open("metadata/tags.cls") as fi:
		taglist = map(lambda x: x[:-1], fi.readlines())

	with open("metadata/mappings.json") as fi:
		mapping = json.loads(fi.read())

	finalTag = map(lambda x: mapping[x], taglist)
	return finalTag



def imageToImageQueries(topk=5):

	finalTag = getNumToTagsMap()
	print(finalTag)

	t = torch.load("savedEmbeddings.pt")
	for i in range(len(t)):
		t[i] = np.concatenate(t[i])

	# Generalize here
	if len(t) == 6:
		imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList = t
	else:
		imgList, audList, imgEmbedList, audEmbedList, vidTagList\
			, audTagList, audioSampleList = t


	print("Loaded embeddings.")

	imgList = bgr2rgb(imgList)
	flag = True

	for i in range(imgEmbedList.shape[0]):
		embed = imgEmbedList[i]
		dist  = ((embed - imgEmbedList)**2).sum(1)
		idx   = dist.argsort()[:topk]
		print(vidTagList[idx])
		plt.clf()
		num_fig = idx.shape[0]
		ax = plt.subplot(1, 3, 1)
		ax.set_title(finalTag[vidTagList[idx[0], 0]])
		plt.axis("off")
		plt.imshow(imgList[idx[0]].transpose(1,2,0))
		for j in range(1, num_fig):
			ax = plt.subplot(2, 3, j+1 + int(j/3))
			ax.set_title(finalTag[vidTagList[idx[j], 0]])
			plt.imshow(imgList[idx[j]].transpose(1,2,0))
			plt.axis("off")

		# plt.tight_layout()
		plt.draw()
		plt.pause(0.001)
		if flag:
			raw_input()
			flag = False
		# res = raw_input("Do you want to save?")
		# if res == "y":
		plt.savefig("results/embed_im_im_{0}.png".format(i))


def crossModalQueries(topk=5, mode1="au", mode2="im"):
	finalTag = getNumToTagsMap()
	print(finalTag)


	for r, di, files in os.walk("Audio_test/"):
		audioFiles = sorted(files)


	t = torch.load("savedEmbeddings.pt")
	for i in range(len(t)):
		t[i] = np.concatenate(t[i])
	imgList, audList, imgEmbedList, audEmbedList, vidTagList\
		, audTagList, audioSampleList = t

	print("Loaded embeddings.")

	imgList = bgr2rgb(imgList)
	flag = True
	print(imgList.shape[0])

	# Open a file and store your queries here
	res = open("results/results_{0}_{1}.txt".format(mode1, mode2), "w+")

	assert(mode1 != mode2)
	try:
		for i in range(imgEmbedList.shape[0]):
			if mode1 == "im":
				embed = imgEmbedList[i]
			else:
				embed = audEmbedList[i]

			# Compute distance
			if mode2 == "im":
				dist = ((embed - imgEmbedList)**2).sum(1)
			else:
				dist = ((embed - audEmbedList)**2).sum(1)

			# Sort arguments
			idx = dist.argsort()[:topk]
			print(vidTagList[idx])
			plt.clf()
			num_fig = idx.shape[0]

			# Actual query
			ax = plt.subplot(2, 3, 1)
			ax.set_title("Query: " + finalTag[vidTagList[i, 0]])
			plt.axis("off")
			plt.imshow(imgList[i].transpose(1,2,0))

			# Top 5 matches
			for j in range(num_fig):
				ax = plt.subplot(2, 3, j+2)
				ax.set_title(finalTag[vidTagList[idx[j], 0]])
				plt.imshow(imgList[idx[j]].transpose(1,2,0))
				plt.axis("off")

			# plt.tight_layout()
			plt.draw()
			plt.pause(0.001)
			if flag:
				raw_input()
				flag = False
			# res = raw_input("Do you want to save?")
			# if res == "y":

			if mode1 == "au":
				res.write(audioFiles[audioSampleList[i, 0]] + "\n")
			else:
				tmpFiles = map(lambda x: audioFiles[audioSampleList[x, 0]], idx)
				line = ", ".join(tmpFiles)
				res.write(line + "\n")

			plt.savefig("results/embed_{0}_{1}_{2}.png".format(mode1, mode2, i))
	except:
		res.close()

	res.close()


if __name__ == "__main__":
	cuda = True
	args = parser.parse_args()
	mode = args.mode

	if mode == "demo":
		demo()
	elif mode == "imageToImageQueries":
		imageToImageQueries()
	elif mode == "crossModalQueries":
		crossModalQueries()
	elif mode == "generateEmbeddingsForVideoAudio":
		generateEmbeddingsForVideoAudio()
	elif mode == "main":
		main(use_cuda=cuda, batch_size=64)
	elif mode == "getVideoEmbeddings":
		getVideoEmbeddings()
	elif mode == "checkValidation":
		checkValidation(use_cuda=cuda, validation="test", batch_size=128, model_name="models/avenet.pt")

	warnings_file.close()
