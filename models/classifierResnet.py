from torch import nn
from utils import utils
import torchvision.models as models
from collections import OrderedDict

class _C(nn.Module):
    """ Classifier class"""
    def __init__(self, input_h_w=28):
        super(_C, self).__init__()     
        self.input_height = input_h_w
        self.input_width = input_h_w
        self.input_dim = 1
        self.output_dim = 11
        self.model =  models.resnet18(pretrained=True)
        
        self.model.fc = nn.Sequential(
            OrderedDict([
                #('fc1', nn.Linear(512,100)),
                #('relu', nn.ReLU()),
                ('fc2', nn.Linear(512, self.output_dim)),
                ('output', nn.LogSoftmax(dim=1))
        ]))
        #self.fc = nn.Linear(512, 1024)
        #utils.initialize_weights(self)

    def forward(self, input_):
        x = self.model(input_)
        #x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        #x = self.fc(x)
        #x = self.last(x)
        
        return x
    
    def forward_partial(self, input_):
        x = self.model.conv1(input_)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.model.fc(x)
        return x
