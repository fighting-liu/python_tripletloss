import numpy as np
import lmdb
import os
import json
import cv2
import requests

import config

def load_json(json_file):
    assert os.path.exists(json_file), \
            'json file not found at: {}'.format(json_file)
    with open(json_file, 'rb') as f:
        data = json.load(f)
    return data

cv_session = requests.Session()
cv_session.trust_env = False
def cv_load_image(in_):
    '''
    Return
        image: opencv format np.array. (C x H x W) in BGR np.uint8
    '''
    if in_[:4] == 'http':
        img_nparr = np.fromstring(cv_session.get(in_).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(in_)
    return img    

def get_image_blob(sample):
    im_blob = []
    for i in range(config.BATCH_SIZE):
        im = cv_load_image(sample[i]['img_path'])
        item_id = str(sample[i]['item_id'])
        bbox = [int(sample[i]['xmin']), int(sample[i]['ymin']), int(sample[i]['width']), int(sample[i]['height'])]
        ## xmin, ymin, w, h --> xmin, ymin, xmax, ymax
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        im = prep_im_for_blob_with_bbox(im, bbox)
        im_blob.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(im_blob)
    return blob, np.ones((config.BATCH_SIZE, 303), dtype=np.int)     

def prep_im_for_blob_with_bbox(im, bbox):
    im = im.astype(np.float32, copy=False)
    im = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    target_size = config.TARGET_SIZE
    pixel_means = np.array([[config.PIXEL_MEANS]])
    im = cv2.resize(im, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    im -= pixel_means
    return im      

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    target_size = config.TARGET_SIZE
    pixel_means = np.array([[config.PIXEL_MEANS]])   
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    return im

class SampleData(object):
    def __init__(self):
        if not config.USE_LMDB:
            assert os.path.exists(config.METADATA_JSON), config.METADATA_JSON
            self._objectid_to_meta = load_json(config.METADATA_JSON)
        self._ap_to_ne_list = load_json(config.TRIPLET_JSON)
        self._ap_keys = self._ap_to_ne_list.keys()
        print 'Num of triplets: ', len(self._ap_keys) 

class FeatureLmdb(object):
    def __init__(self, db, **kwargs):
        self.env = lmdb.open(db, map_size=1e12)

    def put(self, keys, feas, type_='fea'):
        '''Store features in to lmdb backend
        Args:
            keys(list): keys for features
            feas(list): corresponding features
            type_(str): feature type, fea for numpy.ndarray, raw for serialized features
        '''
        with self.env.begin(write=True) as txn:
            for i, k in enumerate(keys):
                if type_ == 'fea':
                    txn.put(k, feas[i].tobytes())
                elif type_ == 'raw':
                    txn.put(k, feas[i])
                else:
                    raise
    def get(self, key):
        '''
        Get feature by key from the lmdb.
        Args:
            key(str): key of the feature
        '''
        with self.env.begin() as txn:
            fea = np.fromstring(txn.get(key), dtype=np.float32)
        return fea

    def get_all(self):
        '''
        Get all features from the lmdb.
        '''
        all_data = {}
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                all_data[k] = np.fromstring(v, dtype=np.float32)
        return all_data

    def get_raws(self):
        '''
        Get all serialized features from the lmdb
        '''
        keys = []
        values = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                keys.append(k)
                values.append(v)
        return keys, values                                        

if __name__ == '__main__':
    sample = sampledata()
