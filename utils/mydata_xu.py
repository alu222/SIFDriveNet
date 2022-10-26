
#6300
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2,os
from scipy import signal
from torchvision.transforms import Compose, Normalize, ToTensor
import scipy
import codecs
class  Mydata(Dataset):
    

    def __init__(self, img_speed_path="train.txt", img_path="train/img/",speed_path="train/speed/", transforms=None,test=False):

        self.img_speed_path=img_speed_path
        self.test=test
        self.img_path = img_path
        self.speed_path = speed_path
        self.transforms = transforms
        self.img_list=[]
        self.speed_list=[]
        self.label_list=[]
       
        with codecs.open(self.img_speed_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                list1=i.split()
                self.img_list.append(list1[0])
                self.speed_list.append(list1[1])
                self.label_list.append(list1[2])
       
        
        self._vid_transform, self._speed_transform = self._get_normalization_transform()


    def _get_normalization_transform(self):
        _vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _speed_transform = Compose([Normalize(mean=[0.0], std=[12.0])])
        return _vid_transform, _speed_transform
    
    def __len__(self):
        
        return len(open(self.img_speed_path,'r').readlines()) #dui  2
    
    def __getitem__(self, idx): 
       
        image_path=os.path.join(self.img_path,self.img_list[idx])
     
        image = cv2.imread(image_path) #cv默认bgr hwc 我们正常读取图片是的通道顺序是h,w,c，但是通过pytorch中的ToTensor()处理之后，读出来的图片数据通道顺序就变成了c,h,w
    
        image = cv2.resize(image, (224,224))
        image = image/255.0
        image = image.transpose(2, 0, 1)

       
        txt_path=os.path.join(self.speed_path,self.speed_list[idx])
     
        samples = []
        with codecs.open("./"+txt_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                samples.append(i)
        samples=list(map(float, samples)) 
        samples=np.array(samples)
        
        frequencies, times, spectrogram =signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
     
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([3])
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        image = self._vid_transform(torch.Tensor(image))
        speed = torch.Tensor(spectrogram.reshape(spec_shape))
        speed = self._speed_transform(speed)

        result=[int(self.label_list[idx])]
        #print("result",result)
        return image, speed, torch.LongTensor(result)






