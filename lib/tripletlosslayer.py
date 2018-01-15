# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""
import caffe
import numpy as np
import yaml
import config


class TripletLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        layer_params = yaml.load(self.param_str)
        self.margin = layer_params['margin']
        self.triplet = config.BATCH_SIZE/3
        self.a = 2.0
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        loss = 0.0      

        ## n*c
        a_batch = bottom[0].data[:self.triplet]
        p_batch = bottom[0].data[self.triplet:2*self.triplet]
        n_batch = bottom[0].data[2*self.triplet:3*self.triplet]

        ##n
        a_p = np.sum((a_batch-p_batch)**2, axis=1, keepdims=False)
        a_n = np.sum((a_batch-n_batch)**2, axis=1, keepdims=False)
        dist = self.margin + a_p - a_n
        self.dist = dist  ##cache for backward

        ## statisticals
        hard_triplet = np.sum(a_p>=a_n)
        no_loss_triplet = np.sum(dist<=0.0)
        semi_hard_triplet = self.triplet - hard_triplet - no_loss_triplet
        print 'Semi-hard Batch(effective):'+str(semi_hard_triplet)+' Hard triplet:'+\
                str(hard_triplet)+' No loss triplet:'+str(no_loss_triplet)

        loss = np.sum(dist[dist>0]) / (2.0*self.triplet)
        top[0].data[...] = loss
    
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            diff = np.zeros_like(bottom[0].data, dtype=np.float32)

            a_batch = bottom[0].data[:self.triplet]
            p_batch = bottom[0].data[self.triplet:2*self.triplet]
            n_batch = bottom[0].data[2*self.triplet:3*self.triplet]

            idx = np.where(self.dist>0)[0]
            # backward for anchor
            diff[idx] = self.a * (n_batch[idx]-p_batch[idx]) / self.triplet
            # backward for positive
            diff[idx+self.triplet] = self.a * (p_batch[idx]-a_batch[idx]) / self.triplet
            # bakcward for negative
            diff[idx+2*self.triplet] = self.a * (a_batch[idx]-n_batch[idx]) / self.triplet

            bottom[0].diff[...] = diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



