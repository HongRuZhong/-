import numpy as np
import torch
import torch.nn  as nn
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.transforms as transform
# from torch.utils.tensorboard import SummaryWriter
# import PyTorch_Tool_Py.classification_train as train_tool
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
import SubPixelShuffle.SubPixelModel as SubPixelModels
import SubPixelShuffle.Load_Data as Load_Data
def Load_Image(ipath):
	img=Image.open(ipath)
	resizeimg=img.resize((32,32))
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
def SubPixelRun():
	opath="D:\\仲鸿儒暂时使用\\RCNN\\FCN_亚像素_超分辨率\\Results\\"
	datapath="D:\\仲鸿儒暂时使用\\RCNN\\FCN_亚像素_超分辨率\\Data\\"
	# writer=SummaryWriter(opath)
	epochs=100
	data_loader=Load_Data.Load_Image(datapath)


	#参数
	net=SubPixelModels.PixelShuffle()
	net.to(device)
	# dataiter = iter(train_loader)
	# hydata, label = dataiter.next()
	# writer.add_graph(net,(train_loader.float(),))

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	lr_img_record=0
	hr_img_record=0
	result=0
	for epoch in range(epochs):
		net.train()
		for index,(data,label) in enumerate(data_loader):
			# print(data.shape,label.shape)
			label=label.float()
			train_loader=data.float()
			train_loader, test_loader=train_loader.to(device),label.to(device)
			output=net(train_loader)
			result=output
			output=output.view(output.size(0),output.size(1)*output.size(2)*output.size(3))
			orig = test_loader.view(test_loader.size(0), test_loader.size(1) * test_loader.size(2) * test_loader.size(3)).float()
			# print(output.shape,label.shape)
			loss = criterion(output, orig)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# writer.add_scalar("loss:",loss,epoch)
			print("epoch:{},loss:{}".format(epoch,loss))
			lr_img_record=train_loader
			hr_img_record=label

	lr_img=lr_img_record[2].cpu().detach().numpy().transpose(1,2,0)
	result_img=result[2].cpu().detach().numpy().transpose(1,2,0)
	hr_img=hr_img_record[2].cpu().detach().numpy().transpose(1,2,0)
	print(lr_img[0:5],result_img[0:5])
	fig,ax=plt.subplots(2,2,figsize=(5,5))
	ax[0,0].imshow(hr_img,cmap=None)
	ax[0, 1].imshow(result_img,cmap=None)
	ax[1, 0].imshow(lr_img,cmap=None)
	plt.show()


	plt.show()
SubPixelRun()
