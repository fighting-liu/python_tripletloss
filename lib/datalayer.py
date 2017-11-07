# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""The data layer used during training to train the network.
   This is a example for online triplet selection
   Each minibatch contains a set of archor-positive pairs, random select negative exemplar
"""

import caffe
import numpy as np
import time
import sys
import os

from utils import SampleData, FeatureLmdb, cv_load_image
import config

class DataLayer(caffe.Layer):
    """Sample data layer used for training."""    
    def _get_next_minibatch(self):  
        triplets = self._triplet
        
        sample = []
        sample_archor = []
        sample_positive = []
        sample_negative = []
        cnt = 0
        while  cnt < triplets:
            if self._index >= len(self._data_container._ap_keys):
                self._index = 0
                np.random.shuffle(self._data_container._ap_keys)
            ## tuple of (anchor_objectid, positive_objectid)
            a_and_p = self._data_container._ap_keys[self._index]    
            ## value of a_and_p, which is a list of negative object_ids 
            negative_list = self._data_container._ap_to_ne_list[a_and_p]
            anchor = a_and_p.strip().split(',')[0]
            positive = a_and_p.strip().split(',')[1]
            negative = negative_list[np.random.randint(len(negative_list))]
            sample_archor.append(anchor)
            sample_positive.append(positive)
            sample_negative.append(negative)              
            self._index +=1 
            cnt += 1       
        ## keep the same data sequence as previous implementation
        sample = sample_archor + sample_positive + sample_negative    
        im_blob, labels_blob = self._get_image_blob(sample)
        #print sample
        blobs = {'data': im_blob,
             'labels': labels_blob}
        return blobs

    def _get_image_blob(self,sample):
        time_1  = time.time()
        try:  
            im_blob = [self.lmdb.get(str(item)).reshape((3, 256, 256)) for item in sample]
        except:
            print sample
        w_beg = np.random.randint(256-224, size=len(im_blob))
        h_beg = np.random.randint(256-224, size=len(im_blob))
        new_im_blob = []
        for idx, img in enumerate(im_blob):
            img = img[:, h_beg[idx]:h_beg[idx]+224, w_beg[idx]:w_beg[idx]+224]
            new_im_blob.append(img)
        im_blob = new_im_blob    
        time_2 = time.time()
        print 'Batch reading time is :{}'.format(time_2-time_1)    
        assert len(im_blob)==self._batch_size 
        im_blob = np.array(im_blob, dtype=np.float32)
        return im_blob, np.ones((self._batch_size, 303), dtype=np.int)

    def set_roidb(self, gpu_id=0):
        assert config.USE_LMDB, 'USE_LMDB flag must be true'
        assert os.path.exists(config.IMG_LMDB_PATH)

        self.lmdb = FeatureLmdb(config.IMG_LMDB_PATH)
        self._batch_size = config.BATCH_SIZE
        assert config.BATCH_SIZE % 3 == 0
        self._triplet = config.BATCH_SIZE/3
        self._data_container = SampleData()      
        if not config.USE_PREFETCH:                
            self._index = 0
            np.random.seed(gpu_id)
            np.random.shuffle(self._data_container._ap_keys)            
     
    def setup(self, bottom, top):
        """Setup the RoIDataLayer.""" 
        batch_size = config.BATCH_SIZE   
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}
        top[0].reshape(batch_size, 3, 224, 224)
        top[1].reshape(batch_size, 303)               
             
    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

if __name__ == '__main__':
    pass
