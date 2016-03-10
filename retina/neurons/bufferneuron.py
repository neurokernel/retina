#!/usr/bin/env python

import numpy as np

import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize

from baseneuron import BaseNeuron

class BufferNeuron(BaseNeuron):
    def __init__(self, n_dict, neuronstate_p, dt, debug=False, LPU_id='buffer',
                 cuda_verbose = False):
        self.num_neurons = len(n_dict['id'])
        self.neuronstate_p = neuronstate_p
        self.debug = debug
        self.LPU_id = LPU_id

        self._I = garray.zeros(self.num_neurons, np.double)

        super(BufferNeuron, self).__init__(n_dict, neuronstate_p, debug,
                                           LPU_id, cuda_verbose)
        self.copy_state = self.get_copy_state()

    @classmethod
    def initneuron(cls, n_dict, neuronstate_p, dt, debug=False, LPU_id=None):
        return cls(n_dict, neuronstate_p, dt, debug, LPU_id)

    def update_internal_state(self, synapse_state_p):
        super(BufferNeuron, self).update_internal_state_default(
            synapse_state_p, self._I)

    def eval(self):
        self.copy_state.prepared_call(
            self.copy_grid, self.copy_block, self._I.gpudata,
            self.neuronstate_p, self.num_neurons)

    def post_run(self):
        super(BufferNeuron, self).post_run()

    def get_copy_state(self):
        src = '''
            #define BLOCKSIZE %(block_size)d
            __global__ void copy_state(%(type)s* in, %(type)s* out, int num_neurons) {
                int bid = blockIdx.x;
                int nid = bid*BLOCKSIZE + threadIdx.x;

                if (nid < num_neurons) {
                    out[nid] = in[nid];
                }
            }

        '''
        dtype = np.double

        self.copy_block = (128, 1, 1)
        self.copy_grid = ((self.num_neurons - 1) // 128 + 1, 1)

        mod = SourceModule(src % {"type": dtype_to_ctype(dtype),
                                  "block_size": self.copy_block[0]},
                           options = self.compile_options)

        func = mod.get_function("copy_state")
        func.prepare('PPi')

        return func

