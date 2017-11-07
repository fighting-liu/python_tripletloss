# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""Train the network."""
import _init_paths
import caffe
from timer import Timer
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf.text_format
import google.protobuf as pb2
import sys
import config
import argparse

from multiprocessing import Process

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """
    def __init__(self, solver_prototxt, gpu_id, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.gpu_id = gpu_id
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(gpu_id)            

    def train_model(self, max_iters):
        """Network training loop."""
        timer = Timer()
        while self.solver.iter < max_iters:
            timer.tic()
            self.solver.step(1)	    
            print 'fc9_1:',sorted(self.solver.net.params['fc9_1'][0].data[0])[0]
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)    

    def getSolver(self):
        return self.solver                      

def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)
    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))

def solve(proto, pretrained_model, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solverW = SolverWrapper(proto, rank, pretrained_model)
    solver = solverW.getSolver()

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    print 'timing:', timing, rank, solver.param.layer_wise_reduce

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)    

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    
    cnt = 0
    while cnt < solver.param.max_iter:
        solver.step(1)
        print 'rank', rank, ' conv5_3:',solver.net.params['conv_stage3_block2_branch2c'][0].data[0][0][0]
        cnt += 1

def train_model_multi_gpu(solver_prototxt, pretrained_model, gpus, timing=False):
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_prototxt, pretrained_model, gpus, timing, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a triplet network')
    parser.add_argument('--gpu', dest='gpus',
                        help='GPU device id to use "0,1,2"',
                        default='0', type=str)
    parser.add_argument('--solver', dest='solver_prototxt',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--timing', dest='timing',
                        help='timeing the training process',
                       default=False, type=bool)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args    


if __name__ == '__main__':
    """Train network.

    e.g python tools/train_multi_gpus.py 
        --gpu 0,1,2,3 
        --solver ./model/model_256_output_layer_data/solver.prototxt 
        --weights /core1/data/home/liuhuawei/clothing_recognition/tripletloss/all_versions/versiton48/tripletloss/resnet77by2_tripletloss_0731_v48_iter_30000.caffemodel
    """
    args = parse_args()
    solver_prototxt = args.solver_prototxt
    pretrained_model = args.pretrained_model
    gpus = map(int, args.gpus.strip().split(','))
    timing = args.timing
    train_model_multi_gpu(solver_prototxt, pretrained_model, gpus, timing)    


