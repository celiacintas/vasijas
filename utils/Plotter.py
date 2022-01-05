from visdom import Visdom
import numpy as np
import torchvision
from torchvision import transforms



# ================================================================== #
#                              VISDOM                                #
# ================================================================== #

class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main', port="8090"):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
        self.scores_window = None
        self.image_window = None
        self.image_val_window = None

    def plot(self, var_name, split_name, x, y, x_label='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def close_window(self, var_name):
        self.viz.close(self.plots[var_name])
        del self.plots[var_name]
        
    def images(self, images, nrow=2):
        if self.image_window != None:
            self.viz.close(self.image_window, env=self.env)
            
        self.image_window = self.viz.images(images, nrow=nrow, env=self.env,
                                            opts=dict(nrow=nrow, title='Images Batch'))
        
        
    def images_val(self, images, nrow=2, title='Images VAL'):
        if self.image_val_window != None:
            self.viz.close(self.image_val_window, env=self.env)
            
        self.image_val_window = self.viz.images(images, nrow=nrow, env=self.env,
                                            opts=dict(nrow=nrow, title=title))
