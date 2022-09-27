import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize

import neurokernel.LPU.utils.curand as curand
from neurokernel.LPU.utils.simpleio import *

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

class RPM_Microvilli(BaseMembraneModel):
    accesses = ['photon', 'I']
    updates = ['V']

    def __init__(self, params_dict, access_buffers, dt, LPU_id=None,
                 debug=False, cuda_verbose = False):
        self.num_microvilli = params_dict['num_microvilli'].get().astype(np.int32) 
        self.num_neurons = int(self.num_microvilli.size)

        self.dt = dt

        self.record_neuron = debug
        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = np.double
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.block_transduction = (256, 1, 1)
        self.grid_transduction = ((self.num_neurons-1)//self.block_transduction[0] + 1, 1) 
        self.block_re_sort = (256, 1, 1)
        self.grid_re_sort = (cuda.Context.get_device().MULTIPROCESSOR_COUNT*5, 1)

        
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self._initialize(params_dict)

    @property
    def maximum_dt_allowed(self):
        return 1e-4

    @property
    def internal_steps(self):
        if self.dt > self.maximum_dt_allowed:
            div = self.dt/self.maximum_dt_allowed
            if np.abs(div - np.round(div)) < 1e-5:
                return int(np.round(div))
            else:
                return int(np.ceil(div))
            #raise ValueError('Simulation time step dt larger than maximum allowed dt of model {}'.format(type(self)))
        else:
            return 1

    @property
    def internal_dt(self):
        return self.dt/self.internal_steps 

    def pre_run(self, update_pointers):
        cuda.memcpy_dtod(int(update_pointers['V']),
                         self.params_dict['init_V'].gpudata,
                         self.params_dict['init_V'].nbytes)

    def _initialize(self, params_dict):
        #self._setup_output()
        self._setup_transduction()
        
        
    def _setup_output(self):
        outputfile = self.LPU_id + '_out'
        if self.record_neuron:
            self.outputfile_I = h5py.File(outputfile+'I.h5', 'w')
            self.outputfile_I.create_dataset(
                '/array', (0, self.num_neurons), dtype = self.dtype,
                maxshape = (None, self.num_neurons))
                
    def _setup_transduction(self):
        self.photons = garray.zeros(self.num_neurons, self.dtype)
        #self.photons = self.params_dict['pre']['photon'].gpudata #where is 'pre'? 
        
        # add x1 & x2
        # while x1 & x2 is of the same size of num_neurons
        self.X = []
        tmp = np.zeros(self.num_neurons, np.double)
        self.X.append(garray.to_gpu(tmp.view(np.double)))
        tmp = np.zeros(self.num_neurons, np.double)
        self.X.append(garray.to_gpu(tmp.view(np.double)))
        
        Xaddress = np.empty(2, np.int64)
        # add the address of x1, x2 array
        for i in range(2):
            Xaddress[i] = int(self.X[i].gpudata)
            
        self.transduction_func = get_transduction_func(self.dtype, Xaddress, self.compile_options)
        self.re_sort_func = get_re_sort_func(self.dtype, self.compile_options)

        

    def run_step(self, update_pointers, st=None):
        
        self.re_sort_func.prepared_async_call(
                self.grid_re_sort, self.block_re_sort, st,
                self.access_buffers['photon'].gpudata,
                self.photons.gpudata,
                self.params_dict['pre']['photon'].gpudata,
                self.params_dict['npre']['photon'].gpudata,
                self.params_dict['cumpre']['photon'].gpudata,
                self.num_neurons)

        # what if no input processor is provided?
        for _ in range(self.internal_steps):
            if self.debug:
                minimum = min(self.photons.get())
                if (minimum < 0):
                    raise ValueError('Inputs to photoreceptor should not '
                                     'be negative, minimum value detected: {}'
                                     .format(minimum))

            # X, photons -> X, V
            self.transduction_func.prepared_async_call(self.grid_transduction, self.block_transduction, st,
              update_pointers['V'], self.photons.gpudata, self.num_neurons, self.internal_dt)

                
def get_transduction_func(dtype, Xaddress, compile_options):
    template = """

extern "C" {
#include "stdio.h"
#include "math.h"

#define b1     215.75 /* x1 synthesis rate*/
#define d1     5 /* x1 diassocaite rate*/
#define b2     2.5335 /* x2 synthesis rate*/
#define d2     4.05775 /* x2 diasscociate rate*/
#define x1_max 1.
#define x2_max 1.
#define r  119.3125 /* r is r */
#define a1 5.382 /* parameters shifting the sigmoid */
#define a2 1.455 /* parameters shifting the sigmoid */


__device__ __constant__ long long int d_X[2];

__device__ double compute_f(float lmbd)
{
    float y = (log10(lmbd) - a1) * a2;
    double f = 1/(1+exp(-y));
    //printf("f = %(temp_type)s", f);
    return f;
}

__device__ double compute_x1 (double x1, double x2, float lmbd, float dt)
{
    double  dx1_dt = b1 * compute_f(lmbd)  * (x1_max-x1-x2) -  r * x2 * x1 - b2 * x1;
    printf("the mother fucker is  %(temp_type)s", dt);
    return x1 + dx1_dt * dt;
}

__device__ double compute_x2 (double x1, double x2, float dt)
{
    double dx2_dt = b2 * x1  - d2 * x2;
    return x2 + dx2_dt * dt;
}

__global__ void
transduction(%(type)s* d_V,  %(type)s* input, int num_neurons, float dt)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_neurons) {
        
        float lambda = input[tid];     // I don't think it will work    
        //printf("lambda = %(temp_type)s", lambda);
        double x1 = ((double*)d_X[0])[tid];
        double x2 = ((double*)d_X[1])[tid];
        double x1_new = compute_x1(x1, x2, lambda, dt);
        double x2_new = compute_x2(x1, x2, dt);
        

        ((double*)d_X[0])[tid] = x1_new;
        ((double*)d_X[1])[tid] = x2_new;
        if (tid == 100)
        {
            //printf("x1 = %(temp_type)s", ((double*)d_X[0])[tid]);

        }
        
        float V = (x1_new * 82 -82);
        d_V[tid] = V;
    }
}
}
"""

    # Used 53 registers, 388 bytes cmem[0], 304 bytes cmem[2]
    # float: Used 35 registers, 380 bytes cmem[0], 96 bytes cmem[2]
    
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype), 
                                 "fletter": 'f' if scalartype == np.float32 else '', 
                                  "temp_type":'%f'}, options = compile_options)
    
    
    '''mod = SourceModule(
        template.format(cool_type = str(dtype_to_ctype(dtype))),
        options = compile_options, no_extern_c = True)'''
    d_X_address, d_X_nbytes = mod.get_global("d_X")
    cuda.memcpy_htod(d_X_address, Xaddress)

    func = mod.get_function('transduction')
    func.prepare('PPif')
    return func
    

# although I have no idea what does the resort do...
def get_re_sort_func(dtype, compile_options):
    template = """

__global__ void
resort(%(type)s* in_photos, %(type)s* out_photons, int* pre, int* npre,
       int* cumpre, int num_neurons)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < num_neurons; i += total_threads)
    {
        if(npre[i])
            out_photons[i] = in_photos[pre[cumpre[i]]];
    }
}
"""
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = compile_options)
    func = mod.get_function('resort')
    func.prepare('PPPPPi')
    return func