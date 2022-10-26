from image_convnet import *
from audio_convnet import *
from AVENet import *
import shutil
import time
import argparse
from torch.optim import *
from torchvision.transforms import *
import warnings
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
from utils.mydata_xu import *
import math
from tqdm import tqdm
from prettytable import PrettyTable

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# warnings_file = open("warning_logs.txt", "w+")
# def customwarn(message, category, filename, lineno, file=None, line=None):
#     warnings_file.write(warnings.formatwarning(message, category, filename, lineno))

# warnings.showwarning = customwarn


choices = ["demo", "main", "test","checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio", \
			"imageToImageQueries", "crossModalQueries"]




class valConfusionMatrix(object):

	def __init__(self, num_classes: int, labels: list):
		self.matrix = np.zeros((num_classes, num_classes))
		self.num_classes = num_classes
		self.labels = labels

	def update(self, preds, labels):
		for p, t in zip(preds, labels):
			self.matrix[p, t] += 1

	def summary(self):

		f1_list=[]

		for i in range(self.num_classes):
			TP = self.matrix[i, i]
			FP = np.sum(self.matrix[i, :]) - TP
			FN = np.sum(self.matrix[:, i]) - TP
			TN = np.sum(self.matrix) - TP - FP - FN
			Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
			Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
			#Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
			F1=round(2*Precision*Recall / (Precision+Recall), 3) if Precision+Recall != 0 else 0.
			
			f1_list.append(F1)
		return f1_list




class testConfusionMatrix(object):
	"""
	注意，如果显示的图像不全，是matplotlib版本问题
	本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
	需要额外安装prettytable库
	"""
	def __init__(self, num_classes: int, labels: list):
		self.matrix = np.zeros((num_classes, num_classes))
		self.num_classes = num_classes
		self.labels = labels

	def update(self, preds, labels):
		for p, t in zip(preds, labels):
			self.matrix[p, t] += 1

	def summary(self):
		# calculate accuracy
		sum_TP = 0
		for i in range(self.num_classes):
			sum_TP += self.matrix[i, i]
		acc = sum_TP / np.sum(self.matrix)
		print("the model accuracy is ", acc)

		# precision, recall, specificity
		table = PrettyTable()
		table.field_names = ["", "Precision", "Recall", "Specificity","F1"]
		for i in range(self.num_classes):
			TP = self.matrix[i, i]
			FP = np.sum(self.matrix[i, :]) - TP
			FN = np.sum(self.matrix[:, i]) - TP
			TN = np.sum(self.matrix) - TP - FP - FN
			Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
			Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
			Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
			F1=round(2*Precision*Recall / (Precision+Recall), 3) if Precision+Recall != 0 else 0.
			table.add_row([self.labels[i], Precision, Recall, Specificity,F1])
		print(table)

	def plot(self):
		matrix = self.matrix
		print(matrix)
		plt.imshow(matrix, cmap=plt.cm.Blues)

		# 设置x轴坐标label
		plt.xticks(range(self.num_classes), self.labels, rotation=45)
		# 设置y轴坐标label
		plt.yticks(range(self.num_classes), self.labels)
		# 显示colorbar
		plt.colorbar()
		plt.xlabel('True Labels')
		plt.ylabel('Predicted Labels')
		plt.title('Confusion matrix')

		# 在图中标注数量/概率信息
		thresh = matrix.max() / 2
		for x in range(self.num_classes):
			for y in range(self.num_classes):
				# 注意这里的matrix[y, x]不是matrix[x, y]
				info = int(matrix[y, x])
				plt.text(x, y, info,
						verticalalignment='center',
						horizontalalignment='center',
						color="white" if info > thresh else "black")
		plt.tight_layout()
		plt.show()


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

checkpoint_dir='/root/total/'
class LossAverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
class AccAverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
	def getacc(self):
		return (self.sum *100) / self.count


# def getimage(video_pathmain="Video1/"):
# 	for r, dirs, files in os.walk(video_pathmain):#正在遍历的文件夹的名字/子文件夹/文件
# 		if len(files) > 0:
# 			video_filesmain = sorted(files)#对文件进行排序
# 			#print("video_files",video_files)
# 			break
			

# 	# list_image1=dict()
# 	list_image1=[]
# 	for i in range(len(video_filesmain)):
# 		list1=getimagelist(k=i,video_filesmain=video_filesmain,video_pathmain=video_pathmain)
# 		# list_image1[i]=list1
# 		list_image1.append(list1)
# 	return list_image1

# def getimagelist(k,video_filesmain,video_pathmain):
# 	vidcap = cv2.VideoCapture(os.path.join(video_pathmain, video_filesmain[k]))
# 	isOpened = vidcap.isOpened()  
# 	framefrequency=math.ceil(vidcap.get(cv2.CAP_PROP_FPS))
# 	#print(framefrequency)
# 	#读帧
# 	list2=[]
# 	i = 0
# 	j=0
# 	while isOpened :
# 		i = i + 1
# 		(success, frame) = vidcap.read()
# 		if not success:
# 			# print("not image")
# 			break
# 		elif (i%framefrequency)==0:
# 			# j=j+1
# 			frame=cv2.resize(frame, (224,224))
# 			frame = frame/255.0
# 			list2.append(frame)
# 			# frame=torch.unsqueeze(torch.Tensor(frame),0)
# 			# if(i==30):
# 			# 	frame_1=frame
# 			# else:
# 			# 	frame_1=torch.cat((frame_1,frame),0)
			
			
# 	#print('图片提取结束') 
# 	vidcap.release() 
# 	# print("j",j)
# 	# print(list1[0].shape)
# 	return list2




# Main function here
def main(use_cuda=True, EPOCHS=200, batch_size=8, model_name="avenet.pt"):
	
	# lossfile = open("losses.txt", "a+")
	# lossfile1 = open("vallosses1.txt", "a+")


	model = getAVENet(use_cuda)
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# if torch.cuda.device_count() > 1:
	# 	print('Lets use', torch.cuda.device_count(), 'GPUs!')
	# 	model = nn.DataParallel(model)
	# model.to(device)


	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	# list_image1=getimage()


	dataset = Mydata(img_speed_path="train.txt", img_path="train/img/",speed_path="train/speed/")
	valdataset = Mydata(img_speed_path="val.txt", img_path="val/img/",speed_path="val/speed/")
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=2)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	# optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
	optim = SGD(model.parameters(), lr=0.25e-3, momentum=0.9, weight_decay=1e-4)
	print("Optimizer loaded.")
	model.train()

	try:
		best_precision = 0
		lowest_loss = 100000
		best_avgf1=0
		best_weightf1=0
		for epoch in range(EPOCHS):
			if(epoch>=100):
				optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
			# Run algo

			train_losses = LossAverageMeter()
			train_acc = AccAverageMeter()
			if (epoch == 0):
				end = time.time()
			for subepoch, (img, aud, out) in enumerate(dataloader):
				if(epoch==0 and subepoch==0):
					print(time.time() - end)
				optim.zero_grad()				
				# Filter the bad ones first
				out = out.squeeze(1)
				idx = (out != 3).numpy().astype(bool)				
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
				# img=img.to(device)
				# aud=aud.to(device)
				o, _, _ = model(img, aud)
				# print(o)
				# print(o.shape, out.shape)
				loss = crossEntropy(o, out)
				train_losses.update(loss.item(),M)
				loss.backward()
				optim.step()
				# Calculate accuracy
				o=F.softmax(o,1)
				_, ind = o.max(1)
				accuracy = (ind.data == out.data).sum()*1.0/M
				train_acc.update((ind.data == out.data).sum()*1.0,M)

				if subepoch%400 == 0:
					print("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d,acc: %f, zongacc: %f" % (epoch, subepoch, train_losses.avg, M, accuracy,train_acc.getacc()))
					with open(file="./losses.txt", mode="a+") as f:
						f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, acc: %f,zongacc: %f"%(epoch, subepoch, train_losses.avg, M,accuracy, train_acc.getacc()))
			print("Epoch: %d, Loss: %f, sum: %d, acc: %f\n"%(epoch, train_losses.avg, train_losses.count, train_acc.getacc()))
			with open(file="./losses.txt", mode="a+") as f:
				f.write("Epoch: %d, Loss: %f, sum: %d, acc: %f\n"%(epoch, train_losses.avg, train_losses.count, train_acc.getacc()))
			val_losses = LossAverageMeter()
			val_acc = AccAverageMeter()
			labels=['Normal','Aggressive','Drowsy']
			valconfusion = valConfusionMatrix(num_classes=3, labels=labels)
			model.eval()
			for sepoch,(img, aud, out) in enumerate(valdataloader):
				out = out.squeeze(1)
				idx = (out != 3).numpy().astype(bool)
				if idx.sum() == 0:
					continue
					# Find the new variables
				img = torch.Tensor(img.numpy()[idx, :, :, :])
				aud = torch.Tensor(aud.numpy()[idx, :, :, :])
				out = torch.LongTensor(out.numpy()[idx])
				img = Variable(img, volatile=True)
				aud = Variable(aud, volatile=True)
				out = Variable(out, volatile=True)
					# print(img.shape, aud.shape, out.shape)
				M = img.shape[0]
				if use_cuda:
					img = img.cuda()
					aud = aud.cuda()
					out = out.cuda()
				with torch.no_grad():
					o, _, _ = model(img, aud)
					valloss = crossEntropy(o, out)
				val_losses.update(valloss.item(),M)
					# Calculate valaccuracy
				o=F.softmax(o,1)
				_, ind = o.max(1)
				valconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
				valaccuracy = (ind.data == out.data).sum()*1.0/M
				val_acc.update((ind.data == out.data).sum()*1.0,M)
				if sepoch%400==0:
					print("Epoch: %d, Sepoch: %d, Valloss: %f, batch_size: %d,  valacc: %f, zongvalacc: %f"%(epoch, sepoch, val_losses.avg, M, valaccuracy,val_acc.getacc()))
					with open(file="./vallosses1.txt", mode="a+") as f:
						f.write("Epoch: %d, Sepoch: %d, Valloss: %f, batch_size: %d,  valacc: %f, zongvalacc: %f"%(epoch, sepoch, val_losses.avg, M, valaccuracy,val_acc.getacc()))
			model.train()
			avgf1=(valconfusion.summary()[0]+valconfusion.summary()[1]+valconfusion.summary()[2])/3.0
			weightnor=0.399
			weightagg=0.257
			weightdrow=0.344
			weightf1=valconfusion.summary()[0]*weightnor+valconfusion.summary()[1]*weightagg+valconfusion.summary()[2]*weightdrow
			print("Epoch: %d, Valloss: %f, sum: %d,  valacc: %f, avgf1: %f, weightf1: %f"%(epoch,  val_losses.avg, val_losses.count, val_acc.getacc(),avgf1,weightf1))
			with open(file="./vallosses1.txt", mode="a+") as f:
				f.write("Epoch: %d,  Valloss: %f, sum: %d,  valacc: %f, avgf1: %f, weightf1: %f\n"%(epoch,val_losses.avg, val_losses.count, val_acc.getacc(),avgf1,weightf1))
			is_best_avgf1=avgf1>best_avgf1
			is_best_weightf1=weightf1>best_weightf1
			is_best = val_acc.getacc() > best_precision
			is_lowest_loss = val_losses.avg < lowest_loss
			best_precision = max(val_acc.getacc(), best_precision)
			lowest_loss = min(val_losses.avg, lowest_loss)
			best_avgf1=max(avgf1,best_avgf1)
			best_weightf1=max(weightf1,best_weightf1)
			with open(file="./vallosses1.txt", mode="a+") as f:
				f.write("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f,best_weightf1: %f"%(epoch, best_precision,lowest_loss,best_avgf1,best_weightf1))
			print('--'*30)
			print("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f,best_weightf1: %f"%(epoch, best_precision,lowest_loss,best_avgf1,best_weightf1))
			print('--' * 30)
		
			save_path = os.path.join(checkpoint_dir,model_name)
			torch.save(model.state_dict(),save_path)
		
			best_path = os.path.join(checkpoint_dir,'best_model.pt')
			if is_best:
				shutil.copyfile(save_path, best_path)
		
			lowest_path = os.path.join(checkpoint_dir, 'lowest_loss.pt')
			if is_lowest_loss:
				shutil.copyfile(save_path, lowest_path)

			best_avgf1_path = os.path.join(checkpoint_dir, 'best_avgf1.pt')
			if is_best_avgf1:
				shutil.copyfile(save_path, best_avgf1_path)
		
			best_weightf1_path = os.path.join(checkpoint_dir, 'best_weightf1.pt')
			if is_best_weightf1:
				shutil.copyfile(save_path, best_weightf1_path)
				

	except Exception as e:
		print(e)
		torch.save(model.state_dict(), "backup"+model_name)
		print("Checkpoint saved and backup.")
	#
	# lossfile.close()
	# lossfile1.close()



def getAVENet(use_cuda):
	model = AVENet()
	# model.fc3.weight.data[0] = -0.1
	# model.fc3.weight.data[1] =  0.1
	# model.fc3.bias.data[0] =   1.0
	# model.fc3.bias.data[1] = - 1.0
	model.fc3.weight.data[0] = -0.7090
	model.fc3.weight.data[1] =  0.7090
	model.fc3.weight.data[2] =  0.7090
	model.fc3.bias.data[0] =   1.2186
	model.fc3.bias.data[1] = - 1.2186
	model.fc3.bias.data[2] = - 1.2186
	if use_cuda:
		model = model.cuda()

	return model



class TestMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n

	def getacc(self):
		return (self.sum *100) / self.count


def test(use_cuda=True, batch_size=8, model_name="lowest_loss.pt"):
	model = getAVENet(use_cuda)
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	testdataset = Mydata(img_speed_path="test.txt", img_path="test/img/",speed_path="test/speed/")
	testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=2)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	test_losses = LossAverageMeter()
	test_acc = TestMeter()
	labels=['Normal','Aggressive','Drowsy']
	testconfusion = testConfusionMatrix(num_classes=3, labels=labels)
	model.eval()
	for sepoch, (img, aud, out) in enumerate(testdataloader):
		out = out.squeeze(1)
		idx = (out != 3).numpy().astype(bool)
		if idx.sum() == 0:
			continue
		# Find the new variables
		img = torch.Tensor(img.numpy()[idx, :, :, :])
		aud = torch.Tensor(aud.numpy()[idx, :, :, :])
		out = torch.LongTensor(out.numpy()[idx])
		img = Variable(img, volatile=True)
		aud = Variable(aud, volatile=True)
		out = Variable(out, volatile=True)
		# print(img.shape, aud.shape, out.shape)
		M = img.shape[0]
		if use_cuda:
			img = img.cuda()
			aud = aud.cuda()
			out = out.cuda()
		with torch.no_grad():
			o, _, _ = model(img, aud)
			valloss = crossEntropy(o, out)
		test_losses.update(valloss.item(), M)
		# Calculate valaccuracy
		o = F.softmax(o, 1)
		_, ind = o.max(1)
		testconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
		x = (ind.data == out.data).sum() * 1.0
		testaccuracy =x / M
		test_acc.update(x, M)
		if sepoch % 300 == 0:
			print("Sepoch: %d, testloss: %f, batch_size: %d,  testacc: %f, zongacc: %f" % (
			sepoch, test_losses.avg, M,testaccuracy, test_acc.getacc()))
			with open(file="./test_lowest_loss.txt", mode="a+") as f:
				f.write(" Sepoch: %d, testloss: %f, batch_size: %d,  testacc: %f, zongacc: %f\n" % (
				sepoch, test_losses.avg, M, testaccuracy, test_acc.getacc()))
	with open(file="./test_lowest_loss.txt", mode="a+") as f:
		f.write("  testloss: %f, batch_size: %d, sum :%d,  testacc: %f\n" % (test_losses.avg, M,test_acc.count, test_acc.getacc()))
	testconfusion.summary()
	testconfusion.plot()





if __name__ == "__main__":
	cuda = True
	args = parser.parse_args()
	mode = args.mode
	# list_image1=getimage()
	if mode == "demo":
		demo()
	elif mode == "main":
		main(use_cuda=cuda, batch_size=16)
	elif mode == "test":
		test(use_cuda=cuda, batch_size=16)

