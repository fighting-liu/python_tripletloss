# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import config

class TripletLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        layer_params = yaml.load(self.param_str)
        self.margin = layer_params['margin']
        self.triplet = config.BATCH_SIZE/3
        
        self.a = 2
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        hard_triplet = 0
        loss = 0.0
        self.no_residual_list = []        

        anchor_minibatch_db = bottom[0].data[:self.triplet]
        positive_minibatch_db = bottom[0].data[self.triplet:2*self.triplet]
        negative_minibatch_db = bottom[0].data[2*self.triplet:3*self.triplet]

        for i in range(self.triplet):
            a = anchor_minibatch_db[i]
            p = positive_minibatch_db[i]
            n = negative_minibatch_db[i]
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p,a_p)
            an = np.dot(a_n,a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist,0.0)
            if ap >= an:
                hard_triplet += 1            
            if i == 0:
                print ('loss:'+' ap:'+str(ap)+' '+'an:'+str(an))
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss
        print 'Semi-hard Batch(effective):'+str(self.triplet-len(self.no_residual_list)-hard_triplet)+' Hard triplet:'+str(hard_triplet)+' No loss triplet:'+str(len(self.no_residual_list)) 
        
        loss = loss/(2*self.triplet)
        top[0].data[...] = loss
    
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            for i in range(self.triplet):
                if not i in self.no_residual_list:
                    x_a = bottom[0].data[i]
                    x_p = bottom[0].data[i+self.triplet]
                    x_n = bottom[0].data[i+2*self.triplet]                                      

                    bottom[0].diff[i] =  self.a*(x_n - x_p) / self.triplet
                    bottom[0].diff[i+self.triplet] =  self.a*(x_p - x_a) / self.triplet 
                    bottom[0].diff[i+2*self.triplet] =  self.a*(x_a - x_n) / self.triplet    
                else:
                    bottom[0].diff[i] = np.zeros(shape(bottom[0].data)[1])
                    bottom[0].diff[i+self.triplet] = np.zeros(shape(bottom[0].data)[1])
                    bottom[0].diff[i+2*self.triplet] = np.zeros(shape(bottom[0].data)[1]) 

    # def backward(self, top, propagate_down, bottom):
    #     if propagate_down[0]:
    #         for i in range(self.triplet):
    #             if not i in self.no_residual_list:
    #                 x_a = bottom[0].data[i]
    #                 x_p = bottom[0].data[i+self.triplet]
    #                 x_n = bottom[0].data[i+2*self.triplet]                                      

    #                 bottom[0].diff[i] =  self.a*(x_n - x_p) / self.triplet
    #                 bottom[0].diff[i+self.triplet] =  self.a*(x_n - x_p) / self.triplet 
    #                 bottom[0].diff[i+2*self.triplet] =  self.a*(x_n - x_p) / self.triplet    
    #             else:
    #                 bottom[0].diff[i] = np.zeros(shape(bottom[0].data)[1])
    #                 bottom[0].diff[i+self.triplet] = np.zeros(shape(bottom[0].data)[1])
    #                 bottom[0].diff[i+2*self.triplet] = np.zeros(shape(bottom[0].data)[1])        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass





