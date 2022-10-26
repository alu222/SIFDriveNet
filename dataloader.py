from __future__ import print_function, division
import os, cv2, json
import torch
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, Normalize, ToTensor

# Get index for each genre
with open("metadata/tags.cls") as fi:
	tags = map(lambda x: x[:-1], fi.readlines())
	tags = dict((x, i) for i, x in enumerate(tags))  #0 x  1  x   2 x


# Function for getting class to video map
# And video to class map
def getMappings(path1="csv/videos1.csv", check_file="metadata/videoToGenre1.json", videoFolder="Video"):
	# Read from files and generate mappings
	if os.path.exists(check_file):
		with open(check_file) as fi:
			vidToGenre, genreToVid = json.loads(fi.read())
		return vidToGenre, genreToVid#vid ('-MhVui65ans', ['/m/042v_gx', '/m/0342h']), ('2AtclfQOjk8', ['/t/dd00004']),
									#gen'/m/03m5k', ['--SQyOb8eS0', '-0VMvk6TlGI', '-0lLVet6szM', '-82swr-YLH4']
	# Else
	vidToGenre = dict()
	genreToVid = dict()
	for path in [path1]:
		# genre to video path
		p = open(path)
		lines = p.readlines()
		for lin in lines: #视频id  starttime endtime  标签ids
			
		
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			#print(words) ['0DLPzsiXXE', '30.000', '40.000', '/m/04rlf', '/m/07qwdck']
			words = words[0:3] + [words[3:]]  #[--0FMNFsVeg,30.000,40.000,[/m/0342h,/m/0342h]]
			#print(words)  ['0DLPzsiXXE', '30.000', '40.000', ['/m/04rlf', '/m/07qwdck']]
			video_id = words[0]#0FMNFsVeg
			#print(video_id) 0DLPzsiXXE

			# Check if video is present in the folder
			if not os.path.exists(os.path.join(videoFolder, "video_" + video_id + ".mp4")):
				
				continue

			vidToGenre[video_id] = words[3]
			# For all genres, add the video to it
			for genre in words[3]:
				genreToVid[genre] = genreToVid.get(genre, []) + [video_id]

	# Save the file
	with open(check_file, "w+") as fi:
		fi.write(json.dumps([vidToGenre, genreToVid]))

	return vidToGenre, genreToVid#两个字典 vidToGenre[0FMNFsVeg]=/m/0342h  genreToVid[/m/0342h]=0FMNFsVeg


def getValMappings(path1="metadata/videos.csv", check_file="metadata/videoToGenreVal.json", videoFolder="Video_val"):
	# Read from files and generate mappings
	
	if os.path.exists(check_file):
		with open(check_file) as fi:
			vidToGenre, genreToVid = json.loads(fi.read())
		return vidToGenre, genreToVid

	# Else
	vidToGenre = dict()
	genreToVid = dict()
	for path in [path1]:
		# genre to video path
		p = open(path)
		lines = p.readlines()
		for lin in lines:
			words = [word.replace("\n","").replace('"', '') for word in lin.replace(" ", "").split(",")]
			words = words[0:3] + [words[3:]]
			video_id = words[0]

			# Check if video is present in the folder
			if not os.path.exists(os.path.join(videoFolder, "video_" + video_id + ".mp4")):
				continue

			vidToGenre[video_id] = words[3]
			# For all genres, add the video to it
			for genre in words[3]:
				genreToVid[genre] = genreToVid.get(genre, []) + [video_id]

	# Save the file
	with open(check_file, "w+") as fi:
		fi.write(json.dumps([vidToGenre, genreToVid]))

	return vidToGenre, genreToVid


## Define custom dataset here
class GetAudioVideoDataset(Dataset):

	def __init__(self, video_path="Video/", audio_path="Audio/", transforms=None, \
			validation=None, return_tags=False, return_audio=False):

		self.video_path = video_path
		self.audio_path = audio_path
		self.transforms = transforms
		self.return_tags = return_tags
		if validation == True or validation == "validation":
			v2g, g2v = getValMappings()
		elif validation == "test":
			v2g, g2v = getValMappings("metadata/videos.csv", "metadata/videosToGenreTest.json", "Video_test")
		else:
			v2g, g2v = getMappings()#两个字典 vidToGenre[0FMNFsVeg]=/m/0342h  genreToVid[/m/0342h]=0FMNFsVeg
									#vid ('-MhVui65ans', ['/m/042v_gx', '/m/0342h']), ('2AtclfQOjk8', ['/t/dd00004']),
									#gen'/m/03m5k', ['--SQyOb8eS0', '-0VMvk6TlGI', '-0lLVet6szM', '-82swr-YLH4']
			#v {'0DLPzsiXXE': ['/m/04rlf', '/m/07qwdck'], '0DLPzsiXX2': ['/m/04rlf', '/m/07qwdck']}
			#g {'/m/04rlf': ['0DLPzsiXXE', '0DLPzsiXX2'], '/m/07qwdck': ['0DLPzsiXXE', '0DLPzsiXX2']}
		self.vidToGenre = v2g#视频id-标签id  -0DLPzsiXXE, 30.000, 40.000, "/m/04rlf,/m/07qwdck"
		self.genreToVid = g2v#标签id-视频id
		self.genreClasses = list(g2v.keys())
		self.sampleRate = 48000
		self.return_audio = return_audio

		# Retrieve list of audio and video files
		for r, dirs, files in os.walk(self.video_path):#正在遍历的文件夹的名字/子文件夹/文件
			if len(files) > 0:
				self.video_files = sorted(files)#对文件进行排序
				#print("self.video_files",self.video_files)#['video_0DLPzsiXX2.mp4', 'video_0DLPzsiXXE.mp4']
				break

		for r, dirs, files in os.walk(self.audio_path):
			if len(files) > 0:
				self.audio_files = sorted(files)
				#print("self.audio_files",self.audio_files)#['audio_0DLPzsiXX2.wav', 'audio_0DLPzsiXXE.wav']
				break

		# Print video and audio files at this point
		# print(self.video_files)
		# print(self.audio_files)

		## Calculate the number of frames and set a length appropriately

		# 40% of the total number of items are positive examples
		# 60% of the total number are negative
		# self.length --> all examples
		fps = 25#应该是25
		time = 10#应该是10
		tot_frames = len(self.video_files)*fps*time #比如3个视频 3*25*10
		#print("tot_frames",tot_frames) 500
		# Frames per video
		self.fps    = fps #25
		self.time   = time#10
		self.fpv    = fps*time#25*10
		self.length = 2*tot_frames#2*3*25*10

		self._vid_transform, self._aud_transform = self._get_normalization_transform()


	def _get_normalization_transform(self):
		_vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		_aud_transform = Compose([Normalize(mean=[0.0], std=[12.0])])

		return _vid_transform, _aud_transform


	def __len__(self):
		# Consider all positive and negative examples
		return self.length #2*3*25*10


	def __getitem__(self, idx):
		print("--------------------------------")
		#print("idx",idx) 0 1 2 3 4 5
		# Given index of item, decide if its positive or negative example, and then 
		#if(idx>=6 ):return (None, None, None)
		if idx >= self.length:
			print("ERROR")
			if self.return_tags:
				if self.return_audio:
					return (None, None, None, None, None, None)
				else:
					return (None, None, None, None, None)
			else:
				return (None, None, None)


		# Positive examples
		if idx < self.length/2:
			#print("Positive")
			video_idx = int(idx/self.fpv)#idx/(25*10) 第几个视频
			#print("video_idx",video_idx) 0
			video_frame_number = idx%self.fpv#idx%(25*10)
			#print("video_frame_number",video_frame_number) 0 1 2 3 4
			frame_time = 500 + (video_frame_number*1000/25)#30应该是25
			#print("frame_time",frame_time)  500 540 580 620 660


			result = [0]
			rate, samples = wav.read(os.path.join(self.audio_path, self.audio_files[video_idx]))#采样率 numpy 数组 1 通道 WAV，数据是 1-D
			# Extract relevant audio file
			#print("samples.shape",samples.shape)  (480375,)
			#ssample=samples[:3]
			#print("ssample",ssample) [-1 -4 -7]
			time  = frame_time/1000.0
			#print("time",time) 0.5 0.54 0.58 0.62 0.66
			# Get video ID
			videoID = self.video_files[video_idx].split("video_")[1].split(".mp4")[0]#视频id  
			#print("videoID",videoID)  0DLPzsiXX2
			vidClasses = self.vidToGenre[videoID]#标签[]
			#print("vidClasses",vidClasses) #['/m/04rlf', '/m/07qwdck']
			vidIndex = tags[vidClasses[0]]#tags[/m/00]  0 1 2 
			#print("vidIndex",vidIndex) 50
			audIndex = vidIndex
			#print("audIndex",audIndex) 50

			# Store the position of audio
			audPos = self.audio_files.index(self.audio_files[video_idx])
			#print("audPos",audPos)  0
		
			


		# Negative examples
		# else:
			
		# 	print("negative")
		# 	video_idx = int((idx-self.length/2)/self.fpv)
		# 	print("video_idx",video_idx)
		# 	video_frame_number = (idx-self.length/2)%self.fpv
		# 	print("video_frame_number",video_frame_number)
		# 	frame_time = 500 + (video_frame_number*1000/30)
		# 	print("frame_time",frame_time)

		# 	result = [1]
		# 	# Check for classes of the video and select the ones not in video
		# 	videoID = self.video_files[video_idx].split("video_")[1].split(".mp4")[0]
		# 	print("videoID",videoID)
		# 	vidClasses = self.vidToGenre[videoID]
		# 	print("vidClasses",vidClasses)
		# 	restClasses = filter(lambda x: x not in vidClasses, self.genreClasses)
		# 	print("restClasses",restClasses)
		# 	randomClass = np.random.choice(restClasses)
		# 	print("randomClass",randomClass)
		# 	randomVideoID = np.random.choice(self.genreToVid[randomClass])
		# 	print("randomVideoID",randomVideoID)
			
		# 	# Store the position of audio
		# 	audPos = self.audio_files.index("audio_" + randomVideoID + ".wav")


		# 	# Read the audio now
		# 	rate, samples = wav.read(os.path.join(self.audio_path, "audio_" + randomVideoID + ".wav"))
		# 	time = (500 + (np.random.randint(self.fpv)*1000/25))/1000.0
		# 	print("time",time)

		# 	# Get video ID
		# 	videoID = self.video_files[video_idx].split("video_")[1].split(".mp4")[0]
		# 	print("videoID",videoID)
		# 	vidClasses = self.vidToGenre[videoID]
		# 	print("vidClasses",vidClasses)
		# 	vidIndex = tags[vidClasses[0]]
		# 	print("vidIndex",vidIndex)
		# 	audIndex = tags[randomClass]
		# 	print("audIndex",audIndex)


		# Extract relevant frame
		#########################
		vidcap = cv2.VideoCapture(os.path.join(self.video_path, self.video_files[video_idx]))
		
		vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time)#视频文件的当前位置（以毫秒为单位）或视频捕获时间戳  当前位置在视频中是多少毫秒 1s=1000ms
	
		image = None
		success = True
		
		if success:
			success, image = vidcap.read()#返回帧
		
			# Some problem with image, return some random stuff
			if image is None:
				if self.return_tags:
					if self.return_audio:
						return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2]) \
								, torch.LongTensor([vidIndex]), torch.LongTensor([audIndex]), torch.LongTensor([-1])
					else:
						return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2]) \
								, torch.LongTensor([vidIndex]), torch.LongTensor([audIndex])

				else:
					return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2])

			image = cv2.resize(image, (224,224))
			image = image/255.0

		else:
			print("FAILURE: Breakpoint 1, video_path = {0}".format(self.video_files[video_idx]))
			return None, None, None, None, None
		##############################
		# Bring the channel to front 
		print("image.shape",image.shape)
		image = image.transpose(2, 0, 1)
		
		start = int(time*48000)-24000
		#print("start",start) 0 1920 3839 5760 7680
		end   = int(time*48000)+24000
		#print("end",end) 48000 49920 51839 53760 55680
		samples = samples[start:end]
		#print("type samples",type(samples)) numpy
		frequencies, times, spectrogram = signal.spectrogram(samples, self.sampleRate, nperseg=512, noverlap=274)
		#print("spectrogram.shape",spectrogram.shape) 257*200
		#print("type spectrogram",type(spectrogram)) 'numpy.ndarray'>
		# Remove bad examples
		if spectrogram.shape != (257, 200):
			if self.return_tags:
				if self.return_audio:
					return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2]) \
							, torch.LongTensor([vidIndex]), torch.LongTensor([audIndex]), torch.LongTensor([audPos])
				else:
					return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2]) \
							, torch.LongTensor([vidIndex]), torch.LongTensor([audIndex])

			else:
				return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([2])

		# Audio
		spectrogram = np.log(spectrogram + 1e-7)
		spec_shape = list(spectrogram.shape)
		spec_shape = tuple([1] + spec_shape)

		image = self._vid_transform(torch.Tensor(image))
		audio = torch.Tensor(spectrogram.reshape(spec_shape))
		audio = self._aud_transform(audio)
		# print(image.shape, audio.shape, result)
		if self.return_tags:
			if self.return_audio:
				return image, audio, torch.LongTensor(result), torch.LongTensor([vidIndex]), torch.LongTensor([audIndex]), torch.LongTensor([audPos])
			else:
				return image, audio, torch.LongTensor(result), torch.LongTensor([vidIndex]), torch.LongTensor([audIndex])

		else:
			return image, audio, torch.LongTensor(result)



if __name__ == "__main__":
	
	dataset = GetAudioVideoDataset()
	dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
	for subepoch, (img, aud, res) in enumerate(dataloader):
		
		print(img.shape, aud.shape, res.shape) #torch.Size([5, 3, 224, 224]) torch.Size([5, 1, 257, 200]) torch.Size([5, 1])
		#print(img.max(), img.min(), aud.max(), aud.min())


	# v,g=getMappings()
	# print("v",v)  {'0DLPzsiXXE': ['/m/04rlf', '/m/07qwdck'], '0DLPzsiXX2': ['/m/04rlf', '/m/07qwdck']}
	# print("g",g)  {'/m/04rlf': ['0DLPzsiXXE', '0DLPzsiXX2'], '/m/07qwdck': ['0DLPzsiXXE', '0DLPzsiXX2']}
