from torch import nn
from utils import utils

class _C(nn.Module):
    """ Classifier class"""
    def __init__(self, input_h_w=28):
        super(_C, self).__init__()     
        self.input_height = input_h_w
        self.input_width = input_h_w
        self.input_dim = 1
        self.output_dim = 11
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.last = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Softmax(),
        )
            
        utils.initialize_weights(self)

    def forward(self, input_):
        x = self.conv(input_)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        x = self.last(x)

        return x
    
    def forward_partial(self, input_):
        x = self.conv(input_)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        return x
