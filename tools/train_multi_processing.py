# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""Train the network."""
import _init_paths
import time
import numpy as np
import sys
from multiprocessing import Process, Queue

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format
import google.protobuf as pb2

from timer import Timer
import config
from utils import SampleData, get_image_blob

import argparse
 
class DataProviderTask(object):
    def __init__(self, data_container):
        super(DataProviderTask, self).__init__()  
        self._data_container = data_container
        self._index = 0
        np.random.shuffle(self._data_container._ap_keys)

        assert config.BATCH_SIZE % 3 == 0
        self._triplet = config.BATCH_SIZE/3 

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
        ## optilonally
        sample = [self._data_container._objectid_to_meta[object_id] for object_id in sample] 
        return sample  
           
class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt,
                 pretrained_model=None, gpu_id=0):
        """Initialize the SolverWrapper."""
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def train_model(self, max_iters, queue):
        """Network training loop."""
        timer = Timer()
        read_time = 0.0
        cnt = 0
        while self.solver.iter < max_iters:
            timer.tic()
            s_time = time.time()
            blobs = queue.get() 
            train_X = blobs['data'].astype(np.float32)
            train_Y = np.ones(train_X.shape[0], dtype=np.float32)
            e_time = time.time()
            read_time += (e_time - s_time)
            cnt += 1
            self.solver.net.set_input_arrays(train_X, train_Y)            
            self.solver.step(1)	    
            print 'conv5_3:',self.solver.net.params['conv_stage3_block2_branch2c'][0].data[0][0][0]
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)   
                print 'Batch reading time is {:.3f}s / batch'.format(read_time / cnt)
                read_time = 0.0
                cnt = 0       

def write_worker(q_out, solver_prototxt, pretrained_model, gpu_id):
    # 
    sw = SolverWrapper(solver_prototxt, pretrained_model, gpu_id) 
    max_iters = sw.solver_param.max_iter 
    print 'Solving...'
    sw.train_model(max_iters, q_out)
    print 'done solving'    
     
def read_worker(q_in, q_out):
    backoff = 0.1
    while True:
        deq = q_in.get()
        if deq is None:
            break
        sample = deq      
        try:
            im_blob, labels_blob = get_image_blob(sample)
        except:
            print 'bad data: ', sample
            continue    
        blobs = {'data': im_blob,
         'labels': labels_blob}

        if q_out.qsize() < 40: 
            q_out.put(blobs) 
            backoff = 0.1
        else:
            time.sleep(backoff)
            backoff *= 2               

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a triplet network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use eg. 0',
                        default=0, type=int)
    parser.add_argument('--process', dest='num_process',
                        help='number of processes to read data',
                        default=5, type=int)    
    parser.add_argument('--solver', dest='solver_prototxt',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    """Train network.

    e.g python tools/train_multi_processing.py 
        --gpu 0  
        --process 1 
        --solver ./model/model_256_output_memory_data/solver.prototxt  
        --weights /core1/data/home/liuhuawei/clothing_recognition/tripletloss/all_versions/versiton48/tripletloss/resnet77by2_tripletloss_0731_v48_iter_30000.caffemodel

    ps: 
    1. batch_size for MemoryData Layer should be the same as config.BATCH_SIZE
    """
    args = parse_args()
    solver_prototxt = args.solver_prototxt
    pretrained_model = args.pretrained_model
    gpu_id = args.gpu_id
    number_of_processes = args.num_process    

    q_out = Queue()
    q_in = [Queue(10) for i in range(number_of_processes)]
    workers = [Process(target=read_worker, args=(q_in[i], q_out)) for i in xrange(number_of_processes)]
    for w in workers:
        w.daemon = True
        w.start() 

    write_process = Process(target=write_worker, args=(q_out, solver_prototxt, pretrained_model, gpu_id))
    write_process.daemon = True
    write_process.start()   

    data_container = SampleData()
    data_prov = DataProviderTask(data_container)

    queue_ind = 0    
    while True:
        sample = data_prov._get_next_minibatch()   
        q_in[queue_ind%number_of_processes].put(sample) 
        queue_ind += 1





