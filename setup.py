"""
Download the VGG16 model and shave off the fully connected layer
before serialising the model to a file.
"""

import torch
import torchvision.models as models
from model.ConvNet import ConvNet

if __name__ == '__main__':
	net = ConvNet()
	model = torch.nn.Sequential(*list(models.vgg16(pretrained=True).children())[0])

	params1 = net.named_parameters()
	params2 = model.named_parameters()

	for ours, theirs in zip(params1, params2):
		theirs[1].data.copy_(ours[1].data)

	torch.save(net.state_dict(), './data/model.pt')
