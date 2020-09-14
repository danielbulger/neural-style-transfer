from torch import nn
import torchvision.models as models


class ConvNet(nn.Module):
	_content_layers = [
		'block4_relu2'
	]

	_style_layers = [
		'block1_relu1',
		'block2_relu1',
		'block3_relu1',
		'block4_relu1',
		'block5_relu1'
	]

	_layers = [
		'block1_conv1',
		'block1_relu1',
		'block1_conv2',
		'block1_relu2',
		'block1_pool',

		'block2_conv1',
		'block2_relu1',
		'block2_conv2',
		'block2_relu2',
		'block2_pool',

		'block3_conv1',
		'block3_relu1',
		'block3_conv2',
		'block3_relu2',
		'block3_conv3',
		'block3_relu3',
		'block3_conv4',
		'block3_relu4',
		'block3_pool',

		'block4_conv1',
		'block4_relu1',
		'block4_conv2',
		'block4_relu2',
		'block4_conv3',
		'block4_relu3',
		'block4_conv4',
		'block4_relu4',
		'block4_pool',

		'block5_conv1',
		'block5_relu1',
		'block5_conv2',
		'block5_relu2',
		'block5_conv3',
		'block5_relu3',
		'block5_conv4',
		'block5_relu4',
		'block5_pool'
	]

	def __init__(self, normalization):
		super(ConvNet, self).__init__()
		model = models.vgg19(pretrained=True).features

		self.normalization = normalization

		self.module_list = nn.ModuleList()

		assert (len(self._layers) == len(model))

		for name, module in zip(self._layers, model):
			if 'pool' not in name:
				self.module_list.add_module(name, module)
			else:
				self.module_list.add_module(name, nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

	def forward(self, x):
		content_layers = []
		style_layers = []

		value = self.normalization(x)

		for name, module in self.module_list.named_children():
			value = module(value)

			if name in self._content_layers:
				content_layers.append(value)

			if name in self._style_layers:
				style_layers.append(value)

		return {
			'output': x,
			'content': content_layers,
			'style': style_layers
		}
