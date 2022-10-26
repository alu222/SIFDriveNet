from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2,os
from scipy import signal
from torchvision.transforms import Compose, Normalize, ToTensor
import scipy
import codecs,math


class  Mydata(Dataset):
    

    def __init__(self, list_image,video_path="Video1/", speed_path="speed_text/",speed_path2="speed_text2/", transforms=None,val=False,test=False):

        self.yuan_speedpath=speed_path
        self.test=test
        self.val=val
        self.video_path = video_path
        self.speed_path = speed_path
        self.transforms = transforms
        self.speed_path2=speed_path2
        self.list_image=list_image
       
        for r, dirs, files in os.walk(self.video_path):#正在遍历的文件夹的名字/子文件夹/文件
            if len(files) > 0:
                self.video_files = sorted(files)#对文件进行排序
                break

        for r, dirs, files in os.walk(self.speed_path):
            if len(files) > 0:
                self.speed_files = sorted(files)
                break

      
     
        self._vid_transform, self._speed_transform = self._get_normalization_transform()

        # self.list_image=dict()
        # for i in range(len(self.video_files)):
        #     list1=self.getimagelist(i)
        #     self.list_image[i]=list1
        # print(len(self.list_image))
        # for i in range(40):
        #     print(len(self.list_image[i]))

        self.train_time=[]
        self.val_time=[]
        self.test_time=[]
        for i in range(len(self.speed_files)):
            x=self.speed_files[i]
            data = np.loadtxt("./"+self.speed_path+x)
            time1 = data[:,0]
            k=(min(int(time1[-1]),len(self.list_image[i]))-1-int(time1[0])-4)//10*6
            k1=(min(int(time1[-1]),len(self.list_image[i]))-1-int(time1[0])-4)//10
            k2=(min(int(time1[-1]),len(self.list_image[i]))-1-int(time1[0])-4)//10*3
            self.train_time.append(k)
            self.val_time.append(k1)
            self.test_time.append(k2)
        # print("train_time",self.train_time)
        # print("val_time",self.val_time)
        # print("test_time",self.test_time)
    

   
    # def getimagelist(self,k):
       
      
    #     vidcap = cv2.VideoCapture(os.path.join(self.video_path, self.video_files[k]))
    #     isOpened = vidcap.isOpened()  
    #     framefrequency=30
    #     #读帧
    #     i = 0
    #     j=0
    #     while isOpened :
    #         i = i + 1
    #         (success, frame) = vidcap.read()
    #         if not success:
    #            # print("not image")
    #             break
    #         elif (i%framefrequency)==0:
    #             # j=j+1
    #             frame=cv2.resize(frame, (224,224))
    #             frame = frame/255.0
    #             frame=torch.unsqueeze(torch.Tensor(frame),0)
    #             if(i==30):
    #                 frame_1=frame
    #             else:
    #                 frame_1=torch.cat((frame_1,frame),0)
                
                
    #     #print('图片提取结束') 
    #     vidcap.release() 
    #     # print("j",j)
    #     # print(list1[0].shape)
    #     return frame_1
    

    def _get_normalization_transform(self):
        _vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _speed_transform = Compose([Normalize(mean=[0.0], std=[12.0])])
        return _vid_transform, _speed_transform
    
    def __len__(self):
        if(self.val):
            return sum(self.val_time)
        elif(self.test):
            return sum(self.test_time)
        else:
            return sum(self.train_time)
    
    def __getitem__(self, idx): #0----1800-1

        if(self.test):
            self.linshi_time=self.test_time
        elif(self.val):
            self.linshi_time=self.val_time
        else:
            self.linshi_time=self.train_time

        if(0<=idx<=self.linshi_time[0]-1):
            video_idx=0
        elif(self.linshi_time[0]<=idx<=sum(self.linshi_time[0:2])-1):
            video_idx=1
        elif(sum(self.linshi_time[0:2])<=idx<=sum(self.linshi_time[0:3])-1):
            video_idx=2
        elif(sum(self.linshi_time[0:3])<=idx<=sum(self.linshi_time[0:4])-1):
            video_idx=3
        elif(sum(self.linshi_time[0:4])<=idx<=sum(self.linshi_time[0:5])-1):
            video_idx=4
        elif(sum(self.linshi_time[0:5])<=idx<=sum(self.linshi_time[0:6])-1):
            video_idx=5
        elif(sum(self.linshi_time[0:6])<=idx<=sum(self.linshi_time[0:7])-1):
            video_idx=6
        elif(sum(self.linshi_time[0:7])<=idx<=sum(self.linshi_time[0:8])-1):
            video_idx=7
        elif(sum(self.linshi_time[0:8])<=idx<=sum(self.linshi_time[0:9])-1):
            video_idx=8
        elif(sum(self.linshi_time[0:9])<=idx<=sum(self.linshi_time[0:10])-1):
            video_idx=9
        elif(sum(self.linshi_time[0:10])<=idx<=sum(self.linshi_time[0:11])-1):
            video_idx=10
        elif(sum(self.linshi_time[0:11])<=idx<=sum(self.linshi_time[0:12])-1):
            video_idx=11
        elif(sum(self.linshi_time[0:12])<=idx<=sum(self.linshi_time[0:13])-1):
            video_idx=12
        elif(sum(self.linshi_time[0:13])<=idx<=sum(self.linshi_time[0:14])-1):
            video_idx=13
        elif(sum(self.linshi_time[0:14])<=idx<=sum(self.linshi_time[0:15])-1):
            video_idx=14
        elif(sum(self.linshi_time[0:15])<=idx<=sum(self.linshi_time[0:16])-1):
            video_idx=15
        elif(sum(self.linshi_time[0:16])<=idx<=sum(self.linshi_time[0:17])-1):
            video_idx=16
        elif(sum(self.linshi_time[0:17])<=idx<=sum(self.linshi_time[0:18])-1):
            video_idx=17
        elif(sum(self.linshi_time[0:18])<=idx<=sum(self.linshi_time[0:19])-1):
            video_idx=18
        elif(sum(self.linshi_time[0:19])<=idx<=sum(self.linshi_time[0:20])-1):
            video_idx=19
        elif(sum(self.linshi_time[0:20])<=idx<=sum(self.linshi_time[0:21])-1):
            video_idx=20
        elif(sum(self.linshi_time[0:21])<=idx<=sum(self.linshi_time[0:22])-1):
            video_idx=21
        elif(sum(self.linshi_time[0:22])<=idx<=sum(self.linshi_time[0:23])-1):
            video_idx=22
        elif(sum(self.linshi_time[0:23])<=idx<=sum(self.linshi_time[0:24])-1):
            video_idx=23
        elif(sum(self.linshi_time[0:24])<=idx<=sum(self.linshi_time[0:25])-1):
            video_idx=24
        elif(sum(self.linshi_time[0:25])<=idx<=sum(self.linshi_time[0:26])-1):
            video_idx=25
        elif(sum(self.linshi_time[0:26])<=idx<=sum(self.linshi_time[0:27])-1):
            video_idx=26
        elif(sum(self.linshi_time[0:27])<=idx<=sum(self.linshi_time[0:28])-1):
            video_idx=27
        elif(sum(self.linshi_time[0:28])<=idx<=sum(self.linshi_time[0:29])-1):
            video_idx=28
        elif(sum(self.linshi_time[0:29])<=idx<=sum(self.linshi_time[0:30])-1):
            video_idx=29
        elif(sum(self.linshi_time[0:30])<=idx<=sum(self.linshi_time[0:31])-1):
            video_idx=30
        elif(sum(self.linshi_time[0:31])<=idx<=sum(self.linshi_time[0:32])-1):
            video_idx=31
        elif(sum(self.linshi_time[0:32])<=idx<=sum(self.linshi_time[0:33])-1):
            video_idx=32
        elif(sum(self.linshi_time[0:33])<=idx<=sum(self.linshi_time[0:34])-1):
            video_idx=33
        elif(sum(self.linshi_time[0:34])<=idx<=sum(self.linshi_time[0:35])-1):
            video_idx=34
        elif(sum(self.linshi_time[0:35])<=idx<=sum(self.linshi_time[0:36])-1):
            video_idx=35
        elif(sum(self.linshi_time[0:36])<=idx<=sum(self.linshi_time[0:37])-1):
            video_idx=36
        elif(sum(self.linshi_time[0:37])<=idx<=sum(self.linshi_time[0:38])-1):
            video_idx=37
        elif(sum(self.linshi_time[0:38])<=idx<=sum(self.linshi_time[0:39])-1):
            video_idx=38
        elif(sum(self.linshi_time[0:39])<=idx<=sum(self.linshi_time[0:40])-1):
            video_idx=39
        
        s1=np.loadtxt("./"+self.speed_path+self.speed_files[video_idx])
        s2 = s1[:,0]
        start=int(s2[0])
        if(self.test):
            imageidx=start+11+(idx-sum(self.test_time[0:video_idx]))//3*10+(idx-sum(self.test_time[0:video_idx]))%3
        elif(self.val):
            imageidx=start+1+10+10*(idx-sum(self.val_time[0:video_idx]))-1
        else:
            imageidx=start+1+3+(idx-sum(self.train_time[0:video_idx]))//6*10+(idx-sum(self.train_time[0:video_idx]))%6
        s=(imageidx-3-start-1)*1260
        e=(imageidx-3-start-1+5)*1260
        image=self.list_image[video_idx][imageidx]   
        #image = image/255.0
        image = image.transpose(2, 0, 1)

        # specdata=np.loadtxt("./"+self.yuan_speedpath+self.speed_path[video_idx])
        # x=specdata[:,0]
        # y=specdata[:,1]
        # f = scipy.interpolate.interp1d(x,y,kind='cubic')
        # x_new = np.linspace(int(x[0])+1, int(x[-1]), 6300*(int(x[-1])-1-int(x[0])))
        # y_new = f(x_new)

        samples=[]
        #specdata=np.loadtxt("./"+self.speed_path2+self.speed_path[video_idx])
        with codecs.open("./"+self.speed_path2+self.speed_files[video_idx], 'r', 'ascii') as infile:
            for i in infile.readlines()[s:e]:
                i = i.strip('\n')
                samples.append(i)
        samples=list(map(float, samples)) 
        samples=np.array(samples)        
       

        # s=(idx-sum(self.time[0:video_idx])-1)*6300
        # e=(idx-sum(self.time[0:video_idx]))*6300
        #samples=y_new[s:e]
        frequencies, times, spectrogram =signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([3])
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        image = self._vid_transform(torch.Tensor(image))
        speed = torch.Tensor(spectrogram.reshape(spec_shape))
        speed = self._speed_transform(speed)

        video_name=self.video_files[video_idx]
        video_name=video_name.split('-')[3]
        if('NORMAL' in video_name):
            style=[0]
        elif('AGGRESSIVE' in video_name):
            style=[1]
        elif('DROWSY' in video_name):
            style=[2]
        #print(style)
        return image, speed, torch.LongTensor(style)



# if __name__ == "__main__":
 
#     list_image1=getimage()
#     train_datasets = Mydata(list_image=list_image1)
    #print(len(train_datasets.list_image))
   
    # train_loader = DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=0)

    # for subepoch, (img, speed, label) in enumerate(train_loader):
    #    # print('subepoch')
    #     #print(subepoch)#一个batchsize
    #     #print('img.shape')
    #     #print(img.shape)
    #     print('label.shape')
    #     #print(label.shape)[batch_size,1]
    #     print(label)#tensor([0, 1])
    #     print('speed.shape')
    #     #print(speed)
    #     print(speed.shape)
    #     label = label.squeeze(1)
    #     #print(label.shape)[batch_size]
    #     idx = (label != 3).numpy().astype(bool)
    #     print(idx.sum())
    #     break
        
        
           



           #file = open("./1.txt", 'w').close()#清空txt
#file = open("./1.txt", 'w+')
#file.write('[1,2]')