import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize

import neurokernel.LPU.utils.curand as curand
from neurokernel.LPU.utils.simpleio import *

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

# yea the suffix 'dumb' means it's not elegent and toooooo engineering
class RPM_Microvilli_dumb(BaseMembraneModel):
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
            self.outputfile_I = h5py.File(outputfile+'I.h5', 'a')
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
            #print(f'the internal dt is {self.internal_dt}')
            #print(f'internal dt type is {type(self.internal_dt)}')
            self.transduction_func.prepared_async_call(self.grid_transduction, self.block_transduction, st,
              update_pointers['V'], self.photons.gpudata, self.num_neurons, self.internal_dt)
            

                
def get_transduction_func(dtype, Xaddress, compile_options):
    template = """
    
extern "C" {
#include "stdio.h"
#include "math.h"

#define b1_LR_0    0.43353 /* x1 synthesis rate*/
#define b1_LR_1    2.374788
#define r_LR_0    -0.32937  /* r is r, r doesn't fucking care */
#define r_LR_1    3.81229
#define b2_LR_0    -0.5436 /* x2 synthesis rate*/
#define b2_LR_1    4.966
#define d2_LR_0    0.05851 /* x2 diasscociate rate*/
#define d2_LR_1    3.79576

/*
#define b1_poly_a  0.1382
#define b1_poly_b  -18.553
#define b1_poly_c  7.5615
#define b1_poly_d  -6.915

#define r_poly_a  -0.6719
#define r_poly_b  9.2384
#define r_poly_c  39.20
#define r_poly_d  55.18

#define b2_poly_a  0.066
#define b2_poly_b  -1.4135
#define b2_poly_c  6.942
#define b2_poly_d  -7.23

#define d2_poly_a  -2.078
#define d2_poly_b  17.402
#define d2_poly_c  -41.04
#define d2_poly_d  32.427 */

#define b1_poly_a  -0.4078
#define b1_poly_b  2.749
#define b1_poly_c  -1.8761

#define r_poly_a  2.1982
#define r_poly_b  -15.791
#define r_poly_c  30.6726

#define b2_poly_a  -0.71954
#define b2_poly_b  4.6346
#define b2_poly_c  -4.8133

#define d2_poly_a  -4.3705
#define d2_poly_b  31.3436
#define d2_poly_c  -43.362

#define x1_max 1.
#define x2_max 1.
#define a1 5.382 /* parameters for shifting the sigmoid */
#define a2 1.455 /* parameters for shifting the sigmoid */


__device__ __constant__ long long int d_X[2];


__device__ float linear_model(double lmbd_log, float a, float b)
{
    // the linear model
    float y = a * lmbd_log + b;
    return y;
}

__device__ float poly_model(double lmbd_log, float a, float b, float c, float d)
{
    // polynomail fitting
    float y = a * pow(lmbd_log, 3) + b * pow(lmbd_log, 2) + c * lmbd_log + d;
    return y;
}

__device__ float poly_model_deg2(double lmbd_log, float a, float b, float c)
{
    // polynomail fitting
    float y = a * pow(lmbd_log, 2) + b * lmbd_log + c;
    return y;
}

__device__ double compute_f(double lmbd_log)
{
    float y = (lmbd_log - a1) * a2;
    double f = 1/(1+exp(-y));
    //printf("f = %(temp_type)s", f);
    return f;
}

__device__ double compute_x1 (double x1, double x2, double lmbd_log, double dt, float b1, float r, float b2)
{   
    double dx1_dt = b1 * compute_f(lmbd_log)  * (x1_max-x1-x2) -  r * x2 * x1 - b2 * x1;
    //printf("the mother fucker is  %(temp_type)s", dx1_dt);
    double x1_new = x1 + dx1_dt * dt;
    if (x1_new > 0) {
        return x1_new;
    }
    else {
        return 0;
    }
}

__device__ double compute_x2 (double x1, double x2, double dt, float b2, float d2)
{
    double dx2_dt = b2 * x1  - d2 * x2;
    double x2_new = x2 + dx2_dt * dt;
    if (x2_new > 0) {
        return x2_new;
    }
    else {
        return 0;
    }
}


__global__ void
transduction(%(type)s* d_V,  %(type)s* input, int num_neurons, float dt)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //float fake_dt = dt;
    if (tid < num_neurons) {
    
    
        double lambda = input[tid];     // I don't think it will work, but it works anyway  
        double lmbd_log = log10(lambda);
        
        //float b1 = linear_model(lmbd_log, b1_LR_0, b1_LR_1) * 50;
        //float r = linear_model(lmbd_log, r_LR_0, r_LR_1) * 50;
        //float b2 = linear_model(lmbd_log, b2_LR_0, b2_LR_1);
        //float d2 = linear_model(lmbd_log, d2_LR_0, d2_LR_1);
        
        //float b1 = poly_model(lmbd_log, b1_poly_a, b1_poly_b, b1_poly_c, b1_poly_d) * 50;
        //float r = poly_model(lmbd_log, r_poly_a, r_poly_b, r_poly_c, r_poly_d) * 50;
        //float b2 = poly_model(lmbd_log, b2_poly_a, b2_poly_b, b2_poly_c, b2_poly_d);
        //float d2 = poly_model(lmbd_log, d2_poly_a, d2_poly_b, d2_poly_c, d2_poly_d);
        
        float b1 = poly_model_deg2(lmbd_log, b1_poly_a, b1_poly_b, b1_poly_c) * 50;
        float r = poly_model_deg2(lmbd_log, r_poly_a, r_poly_b, r_poly_c) * 50;
        float b2 = poly_model_deg2(lmbd_log, b2_poly_a, b2_poly_b, b2_poly_c);
        float d2 = poly_model_deg2(lmbd_log, d2_poly_a, d2_poly_b, d2_poly_c);
        
        double x1 = ((double*)d_X[0])[tid];
        double x2 = ((double*)d_X[1])[tid];
        
        if (tid == 100)
        {
            //printf("lmbd_log = %(temp_type)s", lmbd_log);
            //printf("x1 = %(temp_type)s", x1);
            //printf("b2 = %(temp_type)s", b2);
            //printf("d2 = %(temp_type)s", d2);
            //printf("dt = %(temp_type)s", dt);

        }
        
        double x1_new = compute_x1(x1, x2, lmbd_log, dt, b1, r, b2);
        double x2_new = compute_x2(x1, x2, dt, b2, d2);
        
        //double x1_new = compute_x1(x1, x2, lmbd_log, fake_dt, b1, r, b2);
        //double x2_new = compute_x2(x1, x2, fake_dt, b2, d2);

        ((double*)d_X[0])[tid] = x1_new;
        ((double*)d_X[1])[tid] = x2_new;
        
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
    #func.prepare('PPif')
    func.prepare('PPif') # DO NOT FORGET THIS GUY!!!
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