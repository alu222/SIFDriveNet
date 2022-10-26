# '''Fourtran'''
# '''Author:jAEgerrr'''
# '''2020-03-06'''
# # import matplotlib.pyplot as plt
# # from scipy import signal
# # import numpy as np
# # import pandas as pd
# # from matplotlib import cm
# # import torch
# # import random
# #acc=torch.tensor([1.26,-1.26,1.68,-1.26,1.68,-3.36,-79.07,62.24,-40.37,71.5,59.2,85.1,45.6,78.1,99.9])
# '''acc=np.array([40,50,33,66,78,74,75,85,60,99,80,44,10,22,50,66,30,44,22,5,3,0])
# rate=1
# Block_size=14
# freqs, times, Sxx = signal.spectrogram(acc, fs=rate, 
#                                       nperseg=Block_size, noverlap=0.5*Block_size,
#                                      )
# plt.figure()
# plt.pcolormesh(times, freqs, Sxx,shading='gouraud')

# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.show()'''
# # number=[]
# # for i in range(0, 700):
# #     # 3.生成随机数
# #     num = random.uniform(70,99)
# #     # 4.添加到列表中
# #     number.append(round(num,1))
# # print(number)
# # out=torch.Tensor([0,2,2,2])
# # ind=torch.Tensor([2,1,3,4])
# # print(out.shape)
# # idx = (out != 2).numpy().astype(bool)
# # print(idx)
# # print(idx.sum())
# # print( (ind.data == out.data).sum())
# # n=np.array([0,1,2])
# # print(n.shape)
# # import cv2
# # # img =cv2.imread('./data/train/image1.jpg')#BGR H W C
# # # print(img.shape) 540 960
# # vidcap = cv2.VideoCapture('./Video/video_0DLPzsiXXE.mp4')
		
# # vidcap.set(cv2.CAP_PROP_POS_MSEC, 500)#视频文件的当前位置（以毫秒为单位）或视频捕获时间戳  当前位置在视频中是多少毫秒 1s=1000ms

# # success, image = vidcap.read()#返回帧
# # #print(image.shape) 480 854 3 H W C
# # cv2.imwrite('./Video/0.jpg', image)
# from scipy import signal
# import numpy as np
# # datainfo = open('./RAW_GPS.txt','r')
# # i=0
# # v=[]
# # for line in datainfo:
# #     i=i+1
# #     if i>62:break
# #     speedlist=[]
# #     line = line.strip('\n')
# #     words = line.split()
# #     v.append(float(words[1]))
# # #print(len(v))
# # v=np.array(v)
# # print(v)
# import cv2
# v=np.array([65.2,64.5,63.6,62.2,60.9,61.2,61.7,64.3,66.8,68.9,71.4,73.5,73.5,73.8,75.7,78.0,78.3,79.2,79.2,80.1,80.5,82.0,84.4,86.1,92.7,94.6,99.2,99.2,98.7,99.8,101.1,104.7,104.7,106.3,107.1,108.7,109.3,109.9,110.5,114.8,111.2,110.9,112.3,110.9,111.1,110.4,109.1,107.2,106.9,107.5,107.5,107.5,104.5,107.5,107.6,107.6,107.3,107.7,107.3,108.8,110.1,108.3])

# frequencies, times, spectrogram = signal.spectrogram(v,1, nperseg=3, noverlap=2)#nperseg:int型 窗口长度noverlap:int型 段与段之间的重叠面积 
# print(spectrogram.shape)
# #print(spectrogram)
# #print(type(spectrogram))numpy
# '''  2 60
# [[2.93888889e-01 5.68888889e-01 9.33888889e-01 2.93888889e-01
#   6.72222222e-02 7.20000000e-01 3.29388889e+00 2.80055556e+00
#   2.49388889e+00 2.80055556e+00 9.80000000e-01 5.00000000e-03
#   3.47222222e-01 2.06722222e+00 1.33388889e+00 1.25000000e-01
#   1.80000000e-01 4.50000000e-02 2.68888889e-01 2.93888889e-01
#   1.62000000e+00 2.34722222e+00 5.55555556e+00 1.26672222e+01
#   3.92000000e+00 4.70222222e+00 1.38888889e-02 5.55555556e-04
#   6.80555556e-01 2.13555556e+00 2.88000000e+00 1.42222222e-01
#   8.88888889e-01 5.68888889e-01 8.02222222e-01 1.80000000e-01
#   1.80000000e-01 1.68055556e+00 1.38888889e+00 3.12500000e+00
#   3.55555556e-02 1.08888889e-01 3.75555556e-01 5.00000000e-03
#   4.05000000e-01 1.12500000e+00 9.33888889e-01 0.00000000e+00
#   8.00000000e-02 0.00000000e+00 5.00000000e-01 5.00000000e-01
#   2.06722222e+00 2.22222222e-03 5.00000000e-03 2.22222222e-03
#   8.88888889e-03 2.72222222e-02 1.02722222e+00 3.55555556e-02]
#  [7.54444444e-01 1.75444444e+00 1.73444444e+00 2.14444444e-01
#   2.21111111e-01 5.43000000e+00 6.33444444e+00 4.70777778e+00
#   5.93444444e+00 4.70777778e+00 4.90000000e-01 7.00000000e-02
#   2.88111111e+00 5.00111111e+00 7.34444444e-01 6.70000000e-01
#   9.00000000e-02 6.30000000e-01 2.54444444e-01 1.83444444e+00
#   5.13000000e+00 3.34111111e+00 3.54477778e+01 9.04111111e+00
#   1.78300000e+01 2.35111111e+00 1.94444444e-01 9.07777778e-01
#   1.60777778e+00 1.07877778e+01 1.44000000e+00 1.99111111e+00
#   9.24444444e-01 2.20444444e+00 6.71111111e-01 3.60000000e-01
#   3.60000000e-01 1.47077778e+01 1.04144444e+01 1.63000000e+00
#   1.48777778e+00 1.52444444e+00 2.17777778e-01 3.70000000e-01
#   1.47000000e+00 3.27000000e+00 5.34444444e-01 2.70000000e-01
#   4.00000000e-02 0.00000000e+00 7.00000000e+00 7.00000000e+00
#   1.04111111e+00 1.11111111e-03 7.00000000e-02 1.21111111e-01
#   1.24444444e-01 1.70111111e+00 1.78111111e+00 2.44777778e+00]]
# '''
# #import torch
# #audio = torch.Tensor(cv2.resize(spectrogram,(200,257))).unsqueeze(0)
# #print(audio.shape)torch.Size([1, 257, 200])
# #cv2.resize(spectrogram,(200,257))

# from random import sample
# from matplotlib.cbook import violin_stats
# import numpy as np
# from torch import float64


# data = numpy.loadtxt("./speed_text/20151110175712-16km-D1-NORMAL1-SECONDARY.txt")

# speed = data[:,1]
# time1 = data[:,0]
# # print(speed.shape)(624,)
# # print(type(speed))
# # print(time1.shape)(624,)
# print(time1[0])
# print(time1[-1])


# x="20151110175712-16km-D1-NORMAL1-SECONDARY.txt"
# speed_path="speed_text/"
# data = np.loadtxt("./"+speed_path+x)
# time=[]
# print(data.shape)
# for i in range(len(speed_path)):
#     x=speed_path[i]
#     data = np.loadtxt("./"+speed_path+x)
           
#     time1 = data[:,0]
#     time.append(int(time1[-1])-int(time1[0])-1)


# from scipy import signal
# from scipy.interpolate import interp1d
# speed_path=['20151110175712-16km-D1-NORMAL1-SECONDARY.txt', '20151110180824-16km-D1-NORMAL2-SECONDARY.txt', '20151111123123-25km-D1-NORMAL-MOTORWAY.txt', '20151111125204-24km-D1-AGGRESSIVE-MOTORWAY.txt', '20151111132343-25km-D1-DROWSY-MOTORWAY.txt', '20151111134542-16km-D1-AGGRESSIVE-SECONDARY.txt', '20151111135605-13km-D1-DROWSY-SECONDARY.txt']
# for i in range(7):
#     specdata=np.loadtxt("./speed_text/"+speed_path[i])
#     x=specdata[:,0]
#     y=specdata[:,1]
#     f = interp1d(x,y,kind='cubic')
#     # print(int(x[0])+1)
#     # print(int(x[-1])-1)
#     # print(6300*(int(x[-1])-1-int(x[0])))
#     x_new = np.linspace(int(x[0])+1, int(x[-1])-1, 6300*(int(x[-1])-1-int(x[0])))
#     # print(x_new.shape)
#     y_new = f(x_new)[0:6300]
#     # print(y_new.shape)
#     frequencies, times, spectrogram =signal.spectrogram(y_new, 6300, nperseg=512, noverlap=483)
#     print(spectrogram.shape)
#     break

# video_files=['20151110175712-16km-D1-NORMAL-SECONDARY.mp4', '20151110180824-16km-D1-NORMAL2-SECONDARY.mp4', '20151111123123-25km-D1-NORMAL-MOTORWAY.mp4', '20151111125204-24km-D1-AGGRESSIVE-MOTORWAY.mp4', '20151111132343-25km-D1-DROWSY-MOTORWAY.mp4', '20151111134542-16km-D1-AGGRESSIVE-SECONDARY.mp4', '20151111135605-13km-D1-DROWSY-SECONDARY.mp4']
# x=video_files[0]
# xx=x.split('-')

# if(xx[3]=='NORMAL'):print("1")
# import torch
# out=torch.Tensor([1,0,2,3,2,4])
# idx = (out != 2).numpy().astype(bool)
# out = torch.LongTensor(out.numpy()[idx])
# print(out)


#脚本
# np.set_printoptions(threshold=np.inf)
# from scipy import signal
# import os
# from scipy.interpolate import interp1d
# import numpy as np
# #speed_path=['20151110175712-16km-D1-NORMAL-SECONDARY.txt', '20151111123124-25km-D1-NORMAL-MOTORWAY.txt', '20151111125204-24km-D1-AGGRESSIVE-MOTORWAY.txt', '20151111132343-25km-D1-DROWSY-MOTORWAY.txt', '20151111134542-16km-D1-AGGRESSIVE-SECONDARY.txt', '20151111135605-13km-D1-DROWSY-SECONDARY.txt']
# for r, dirs, files in os.walk("./speed_text/"):
#     if len(files) > 0:
#         speed_path = sorted(files)


# for i in range(len(speed_path)):
#     # file = open("./speed_text1/"+speed_path[i], 'w').close()
#     # file = open("./speed_text2/"+speed_path[i], 'w').close()
#     specdata=np.loadtxt("./speed_text/"+speed_path[i])
#     x=specdata[:,0]
#     y=specdata[:,1]
#     f = interp1d(x,y,kind='cubic')
#     x_new = np.linspace(int(x[0])+1, int(x[-1]), 1260*(int(x[-1])-1-int(x[0])))
#     print(x_new.shape)
#     xx=x_new
#     xx=str(list(xx))
#     xx=xx.replace(' ','')
#     xx=xx.strip('[')
#     xx=xx.strip(']')
#     xx=xx.replace(',','\n')
#     file = open("./speed_text3/"+speed_path[i], 'w')
#     file.write(xx)

#      #(3918600,) (3937500,) (4630500,) (5903100,) (3231900,) (3137400,) val (5865300,) (5367600,)(5827500,)
#     y_new = f(x_new)
#     print(y_new.shape)
#     yy=y_new
#     yy=str(list(yy))
#     yy=yy.replace(' ','')
#     yy=yy.strip('[')
#     yy=yy.strip(']')  
#     yy=yy.replace(',','\n')

#     file1 = open("./speed_text4/"+speed_path[i], 'w')
#     file1.write(yy)

















# s1=np.loadtxt("./speed_text1/20151110175712-16km-D1-NORMAL1-SECONDARY.txt")
# print(type(s1))

# datainfo = open("./speed_text2/20151110175712-16km-D1-NORMAL1-SECONDARY.txt",'r')
# x=0
# speedlist=[]
# for line in datainfo:
#     x=x+1
#     line = line.strip('\n')
    
#     speedlist.append(line)
#     if(x>6):break
# speedlist=list(map(float, speedlist)) 
# print(speedlist)

#import codecs
# lines = []
# with codecs.open("./speed_text2/20151110175712-16km-D1-NORMAL1-SECONDARY.txt", 'r', 'ascii') as infile:
#     for i in infile.readlines()[0:3]:
#         i = i.strip('\n')
#         lines.append(i)
# lines=list(map(float, lines)) 
# # print(lines)
# from scipy import signal
# import os
# path="./speed_text/"
# speed_path2="speed_text2/"
# for r, dirs, files in os.walk(path):
#     speed_path = sorted(files)

# sum=0
# ss=0
# time=[622, 625, 735, 937, 513, 498]

# s1=np.loadtxt("./"+speed_path2+speed_path[5])
# for s in range(498):
#     samples=s1[6300*s:(s+1)*6300]
#     if(s==0):print(samples[0])
#     frequencies, times, spectrogram =signal.spectrogram(samples, 6300, nperseg=512, noverlap=483)
#     if(s%100==0):print(spectrogram.shape)
#     if spectrogram.shape != (257, 200):sum=sum+1
#     ss=ss+1
# print(sum)
# # print(ss)
# import os
# from scipy import signal
# import codecs
# import numpy as np
# # speed_path=[]

# # for r, dirs, files in os.walk("speed_text/"):
# #     print(0)
# #     if len(files) > 0:
# #         speed_path = sorted(files)
# #     else : print(0)
# # print(speed_path)

# #
# ss=0
# sum=0
# a=0
# # # #with codecs.open("./speed_text4/"+speed_path[0], 'r', 'utf-8') as infile:  4,403  4,403
# for s in range(304,404):
#         samples = []
#         # print(1260*(s-4))
#         # print((s+1)*1260)
#         with codecs.open("./speed_text2/20151111135605-13km-D1-DROWSY-SECONDARY.txt", 'r', 'ascii') as infile:
#             for i in infile.readlines()[1260*(s-4):(s+1)*1260]:
#                 i = i.strip('\n')
#                 #print(i)
#                 if a==0:
#                     print(i)
#                     a=a+1
#                     print(a)
#                 samples.append(i)

#         samples=list(map(float, samples))
#         samples=np.array(samples)
#             # print(samples.size)
#             # print(samples[0])
#         #print(samples.size)
#         frequencies, times, spectrogram =signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
#             #if(s%100==0):print(spectrogram.shape)
#         if spectrogram.shape != (257, 200):
#             # if(s==305):print(spectrogram.shape)
#             sum=sum+1
#         ss=ss+1


# print(sum)
# print(ss)














# import torch
# np.set_printoptions(precision=14)
# s1=np.loadtxt("./speed_text2/20151110175712-16km-D1-NORMAL-SECONDARY.txt")

# a=(s1[0:2])

# print(a)
# a=list(s1[0:2])
# print(a)
# print(np.array(a,dtype=float))



#提取图像脚本
# import cv2
# import numpy as np

# def save_image(image,addr,num):
#     address = addr + str(num)+ '.jpg'
#     cv2.imwrite(address,image)

# videoCapture = cv2.VideoCapture('/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151111135612-13km-D1-DROWSY-SECONDARY/20151111135605-13km-D1-DROWSY-SECONDARY.mp4')
# isOpened = videoCapture.isOpened()  
# print(isOpened)
# framefrequency=30
# #读帧
# i = 0
# j=0
# while isOpened :
#     i = i + 1
#     (success, frame) = videoCapture.read()
#     if not success:
#         print("not image")
#         break
#     elif (i%framefrequency)==0:
#         j=j+1
#         save_image(frame,'/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151111135612-13km-D1-DROWSY-SECONDARY/image/',j)
#         print('save image:',j)
# print('图片提取结束') 
# videoCapture.release() 



#生成speedtxt脚本
# import numpy as np
# from scipy import signal
# from scipy.interpolate import interp1d
# specdata=np.loadtxt("/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151111135612-13km-D1-DROWSY-SECONDARY/RAW_GPS.txt")
# x=specdata[:,0]
# print((int(x[-1])-1-int(x[0])))
# y=specdata[:,1]
# f = interp1d(x,y,kind='cubic')
# x_new = np.linspace(int(x[0])+1, int(x[-1]), 6300*(int(x[-1])-1-int(x[0])))
# print(x_new.shape)
#     #622  862 735 937 513 498
#      #(3918600,) (5430600,) 4630500 5903100 3231900 3137400
# y_new = f(x_new)
# print(y_new.shape)
# k=int(x[0])+1
# for i in range((int(x[-1])-1-int(x[0]))):
#     yy=y_new[6300*i:6300*(i+1)]
#     yy=str(list(yy))
#     yy=yy.replace(' ','')
#     yy=yy.strip('[')
#     yy=yy.strip(']')  
#     yy=yy.replace(',','\n')
#     file1 = open("/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151111135612-13km-D1-DROWSY-SECONDARY/speedtxt/"+str(k)+".txt", 'w')
#     file1.write(yy)
#     k=k+1

#复制图片到指定文件夹
 
# import glob
 
# import shutil
# import os
 
# filePath='/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151110175712-16km-D1-NORMAL1-SECONDARY/image'
# newFilePath='/home/lulu/Documents/UAH-DRIVESET-v1/D1/20151110175712-16km-D1-NORMAL1-SECONDARY/trainimg'
 
# s=0

# for root, dirs, files in os.walk(filePath):
#     for i in range(len(files)):
#         print(files[i])
#         file_path = filePath+'/'+files[i]  

#         new_file_path = newFilePath+ '/'+ files[i]  
#         shutil.copy(file_path,new_file_path)  
#         if(i==0):break
from matplotlib import pyplot as plt
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集、验证集loss值对比图")
    plt.show()

loss_train=[0.23385616270204385, 0.10780503012239934, 0.11894822880625724,0.0921548,0.0723541,0.036748,0.0158743,0.0352146]
loss_val=[0.05363623380661011, 0.04302590876817703, 0.0573239572346211,0.048498,0.037845,0.0458712,0.02587,0.0112304]
acc_train=[0.54, 0.7933333333333333, 0.8,0.84,0.89,0.92,0.95,0.98]
acc_val=[0.9, 0.88, 0.84,0.89,0.92,0.95,0.955,0.97]
matplot_loss(loss_train,loss_val)