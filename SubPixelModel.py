import torch.nn as nn

class PixelShuffle(nn.Module): #亚像素超分辨率的实现
	def __init__(self):
		super(PixelShuffle,self).__init__()
		self.conv_1=nn.Conv2d(3,64,kernel_size=5,padding=2)
		self.activate=nn.Tanh()
		self.conv_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
		self.conv_3=nn.Conv2d(32,12,kernel_size=3,padding=1)
		self.pixelconv=nn.PixelShuffle(2)

	def forward(self, input):
		x=self.activate(self.conv_1(input))
		x=self.activate(self.conv_2(x))
		x = self.activate(self.conv_3(x))
		x=self.pixelconv(x)
		return x