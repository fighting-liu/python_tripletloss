# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""

import caffe
import numpy as np
import yaml


class Norm2Layer(caffe.Layer):
    """norm2 layer used for L2 normalization."""

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        top[0].reshape(bottom[0].num, bottom[0].data.shape[1])
        ## for numerical stability
        self.pos_eps = 1e-12

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        x_trans_x_pow = (np.sum((bottom[0].data*bottom[0].data), axis=1, keepdims=True))**(0.5)   ## m*1
        y = bottom[0].data / (x_trans_x_pow+self.pos_eps)

        self.x_trans_x_pow = x_trans_x_pow
        self.y = y
        top[0].data[...] = y.astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """This layer does not need to backward propogate gradient"""
        if propagate_down[0]:
            bottom[0].diff[...] =  (top[0].diff - self.y * np.sum(top[0].diff*self.y, axis=1, keepdims=True)) / (self.x_trans_x_pow+self.pos_eps)         

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward.
        bottom[0]: n*c or n*c*1*1 --> n*c
        """
        bottom[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])
        top[0].reshape(bottom[0].num, bottom[0].data.shape[1])
