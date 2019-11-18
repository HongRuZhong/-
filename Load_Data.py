import PIL.Image as Image
import numpy as np
import os
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader

imgpath="D:\\仲鸿儒暂时使用\\RCNN\\FCN_亚像素_超分辨率\\Data\\"

class DataToTorch(Dataset):
	def __init__(self, data,label):
		# 定义好 image 的路径

		self.data, self.label = data, label

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return len(self.data)

def Load_Image(ipath):
	filepath_list=os.listdir(ipath)
	img_attri_list=[]
	img_label_list=[]
	for x in filepath_list:
		img=Image.open(os.path.join(ipath,x))
		img_resize=img.resize((int(img.size[0]/2),int(img.size[1]/2)))
		img_attri_list.append(np.array(img_resize))
		img_label_list.append(np.array(img))
	img_attri=np.array(img_attri_list)
	img_label=np.array(img_label_list)
	# img_attri=transform.ToTensor()(img_attri)
	img_attri_list.clear()
	img_label_list.clear()
	for x in range(len(img_attri)):
		attri=transform.ToTensor()(img_attri[x]).numpy()
		label=transform.ToTensor()(img_label[x]).numpy()
		# print(img_attri[x].shape)
		# attri=img_attri[x].transpose(2,0,1)
		# label=img_label[x].transpose(2,0,1)
		img_attri_list.append(attri)
		img_label_list.append(label)
	img_train_loader=np.array(img_attri_list)
	img_label_loader=np.array(img_label_list)
	img_Data=DataToTorch(img_train_loader,img_label_loader)

	img_Data_loader=DataLoader(img_Data,batch_size=32,shuffle=False,num_workers=0)
	# img_label_loader=DataLoader(img_label_loader,batch_size=32,shuffle=False,num_workers=4)
	return img_Data_loader

	# Image.open()

aa=Load_Image(imgpath)
# for x in aa:
# 	print(x.shape)