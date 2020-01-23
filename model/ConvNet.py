from torch import nn
import torchvision.models as models


class ConvNet(nn.Module):

	def __init__(self):
		super(ConvNet, self).__init__()
		model = models.vgg16(pretrained=True).features

		self.block1 = nn.Sequential()
		self.block2 = nn.Sequential()
		self.block3 = nn.Sequential()
		self.block4 = nn.Sequential()
		self.block5 = nn.Sequential()

		# Get from layer1-conv1 to layer1-relu2
		for x in range(4):
			self.block1.add_module(str(x), model[x])

		# Get from layer1-pool1 to layer2-relu2
		for x in range(4, 9):
			self.block2.add_module(str(x), model[x])

		# Get from layer2-pool1 to layer3-relu3
		for x in range(7, 16):
			self.block3.add_module(str(x), model[x])

		# Get from layer4-pool1 to layer4-relu3
		for x in range(16, 23):
			self.block4.add_module(str(x), model[x])

		# Get from layer5-pool1 to layer5-relu3
		for x in range(23, 30):
			self.block5.add_module(str(x), model[x])

	def forward(self, x):
		content_layers = []
		style_layers = []

		value = self.block1(x)
		style_layers.append(value)

		value = self.block2(value)
		style_layers.append(value)

		value = self.block3(value)
		style_layers.append(value)

		value = self.block4(value)
		style_layers.append(value)
		content_layers.append(value)

		value = self.block5(value)
		style_layers.append(value)

		return {
			'output': x,
			'content': content_layers,
			'style': style_layers
		}
