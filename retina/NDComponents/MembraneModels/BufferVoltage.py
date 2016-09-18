
import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

class BufferVoltage(BaseMembraneModel):
    updates = ['V']
    accesses = ['V']
    def __init__(self, params_dict, access_buffers, dt, LPU_id=None,
                 debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []
                
        self.dt = np.double(dt)
        self.debug = debug
        self.dtype = np.double

        self.LPU_id = LPU_id

        self.params_dict = params_dict
        
        self.num_neurons = params_dict['pre']['V'].size
        self.access_buffers = access_buffers
        self.update = get_re_sort_func(self.dtype, self.compile_options)
        
        self.block_re_sort = (256, 1, 1)
        self.grid_re_sort = (cuda.Context.get_device().MULTIPROCESSOR_COUNT*5, 1)

    def run_step(self, update_pointers, st=None):
        self.update.prepared_async_call(
            self.grid_re_sort, self.block_re_sort, st,
            self.access_buffers['V'].gpudata,
            update_pointers['V'],
            self.params_dict['pre']['V'].gpudata,
            self.params_dict['npre']['V'].gpudata,
            self.params_dict['cumpre']['V'].gpudata,
            self.num_neurons)

def get_re_sort_func(dtype, compile_options):
    template = """

__global__ void
resort(%(type)s* in_v, %(type)s* out_v, int* pre, int* npre,
       int* cumpre, int num_neurons)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < num_neurons; i += total_threads)
    {
        if(npre[i])
            out_v[i] = in_v[pre[cumpre[i]]];
    }
}
"""
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = compile_options)
    func = mod.get_function('resort')
    func.prepare('PPPPPi')
    return func
