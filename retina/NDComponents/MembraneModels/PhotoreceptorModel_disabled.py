import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize

import neurokernel.LPU.utils.curand as curand
from neurokernel.LPU.utils.simpleio import *

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

class PhotoreceptorModel_disabled(BaseMembraneModel):
    accesses = ['photon', 'I']
    updates = ['V']

    def __init__(self, params_dict, access_buffers, dt, LPU_id=None,
                 debug=False, cuda_verbose = False):
        self.num_microvilli = params_dict['num_microvilli'].get().astype(np.int32)
        self.num_neurons = self.num_microvilli.size

        self.dt = dt

        # self.multiple = int(self.dt/self.run_dt)
        # assert(self.multiple * self.run_dt == self.dt)

        self.record_neuron = debug
        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = np.double
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.block_transduction = (128, 1, 1)
        self.grid_transduction = (cuda.Context.get_device().MULTIPROCESSOR_COUNT*9, 1)
        self.block_sum = (256,1,1)
        self.grid_sum = (self.num_neurons,1)
        self.block_re_sort = (256, 1, 1)
        self.grid_re_sort = (cuda.Context.get_device().MULTIPROCESSOR_COUNT*5, 1)
        self.block_hh = (256, 1, 1)
        self.grid_hh = ((self.num_neurons-1)//self.block_hh[0] + 1, 1)
        self.block_state = (32, 32, 1)
        self.grid_state = ((self.num_neurons-1)//self.block_state[0] + 1, 1)

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
        self._setup_output()
        self._setup_transduction()
        self._setup_hh()

    def _setup_output(self):
        outputfile = self.LPU_id + '_out'
        if self.record_neuron:
            self.outputfile_I = h5py.File(outputfile+'I.h5', 'w')
            self.outputfile_I.create_dataset(
                '/array', (0, self.num_neurons), dtype = self.dtype,
                maxshape = (None, self.num_neurons))
            
    def _setup_transduction(self, seed = 0):
        self.photons = garray.zeros(self.num_neurons, self.dtype)

        # setup RNG
        self.randState = curand.curand_setup(
            self.block_transduction[0]*self.grid_transduction[0], seed)

        # using microvilli as single unite in the transduction kernel
        # therefore, we need to figure out which neuron each microvillus
        # belongs to, and from where to where we should sum up the current.
        self.cum_microvilli = np.hstack((0, np.cumsum(self.num_microvilli)))
        self.total_microvilli = self.cum_microvilli[-1]
        tmp = np.zeros(self.total_microvilli, np.uint16)
        tmp[self.cum_microvilli[1:-1]] = 1
        self.microvilli_ind = np.cumsum(tmp).astype(np.uint16)
        #self.d_num_microvilli = garray.to_gpu(self.num_microvilli)
        self.d_num_microvilli = self.params_dict['num_microvilli']

        self.count = garray.empty(1, np.int32)

        self.d_cum_microvilli = garray.to_gpu(self.cum_microvilli.astype(np.int32))
        self.d_microvilli_ind = garray.to_gpu(self.microvilli_ind.astype(np.uint16))

        self.X = []
        tmp = np.zeros(self.total_microvilli*2, np.uint16)
        tmp[::2] = 50
        # variables G, Gstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros(self.total_microvilli*2, np.uint16)
        # variables PLCstar, Dstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros(self.total_microvilli*2, np.uint16)
        # variables Cstar, Tstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros(self.total_microvilli, np.uint16)
        # variables Mstar
        self.X.append(garray.to_gpu(tmp))


        Xaddress = np.empty(5, np.int64)
        for i in range(4):
            Xaddress[i] = int(self.X[i].gpudata)
        Xaddress[4] = int(self.d_microvilli_ind.gpudata)

        change_ind1 = np.asarray([1, 1, 2, 3, 3, 2, 5, 4, 5, 5, 7, 6, 6, 1],
                                 np.int32) - 1
        change_ind2 = np.asarray([1, 1, 3, 4, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1],
                                 np.int32) - 1
        change1 = np.asarray([0, -1, -1, -1, -1, 1, 1, -1, -1, -2, -1, 1, -1, 1],
                             np.int32)
        change2 = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             np.int32)

        self.transduction_func = get_transduction_func(
            self.dtype, self.block_transduction[0], Xaddress,
            change_ind1, change_ind2,
            change1, change2, self.compile_options)

        self.re_sort_func = get_re_sort_func(
            self.dtype, self.compile_options)

        self.ns = garray.zeros(self.num_neurons, self.dtype) + 1
        self.update_ns_func = get_update_ns_func(self.dtype, self.compile_options)

    def _setup_hh(self):
        self.I = garray.zeros(self.num_neurons, self.dtype)
        self.I_fb = garray.zeros(self.num_neurons, self.dtype)

        self.hhx = [garray.empty(self.num_neurons, self.dtype)
                    for _ in range(5)]

        # mV
#        V_init = np.empty(self.num_neurons, dtype=np.double)
#        V_init.fill(-0.0819925*1000)
        #cuda.memcpy_htod(int(self.V_p), V_init)

        self.hhx[0].fill(0.2184)
        self.hhx[1].fill(0.9653)
        self.hhx[2].fill(0.0117)
        self.hhx[3].fill(0.9998)
        self.hhx[4].fill(0.0017)

        self.sum_current_func = get_sum_current_func(self.dtype,
                                                     self.block_sum[0],
                                                     self.compile_options)
        self.hh_func = get_hh_func(self.dtype, self.compile_options)

    def run_step(self, update_pointers, st=None):
        self.I_fb.fill(0)
        if self.params_dict['pre']['I'].size > 0:
            self.sum_in_variable('I', self.I_fb)

        # what if no input processor is provided?
        self.re_sort_func.prepared_async_call(
                self.grid_re_sort, self.block_re_sort, st,
                self.access_buffers['photon'].gpudata,
                self.photons.gpudata,
                self.params_dict['pre']['photon'].gpudata,
                self.params_dict['npre']['photon'].gpudata,
                self.params_dict['cumpre']['photon'].gpudata,
                self.num_neurons)

        for _ in range(self.internal_steps):
            if self.debug:
                minimum = min(self.photons.get())
                if (minimum < 0):
                    raise ValueError('Inputs to photoreceptor should not '
                                     'be negative, minimum value detected: {}'
                                     .format(minimum))

            # reset warp counter
            self.count.fill(0)

            # X, V, ns, photons -> X
            self.transduction_func.prepared_async_call(
                self.grid_transduction, self.block_transduction, st,
                self.randState.gpudata, self.internal_dt,
                update_pointers['V'], self.ns.gpudata,
                self.photons.gpudata,
                self.d_num_microvilli.gpudata,
                self.total_microvilli, self.count.gpudata)

            # X, V, I_fb -> I
            self.sum_current_func.prepared_async_call(
                self.grid_sum, self.block_sum, st,
                self.X[2].gpudata, self.d_num_microvilli.gpudata,
                self.d_cum_microvilli.gpudata,
                update_pointers['V'], self.I.gpudata, self.I_fb.gpudata)

            # hhX, I -> hhX, V
            self.hh_func.prepared_async_call(
                self.grid_hh, self.block_hh, st,
                self.I.gpudata, update_pointers['V'], self.hhx[0].gpudata,
                self.hhx[1].gpudata, self.hhx[2].gpudata, self.hhx[3].gpudata,
                self.hhx[4].gpudata, self.num_neurons, self.internal_dt/10, 10)

            self.update_ns_func.prepared_async_call(
                ( (self.num_neurons - 1) // 128 + 1, 1), (128, 1, 1), st,
                self.ns.gpudata, self.num_neurons, update_pointers['V'], self.internal_dt)



def get_update_ns_func(dtype, compile_options):

    template_run = """

#include "curand_kernel.h"

extern "C" {
#include "stdio.h"

#define BLOCK_SIZE %(block_size)d
#define LA 0.5

__device__ __constant__ long long int d_X[5];
__device__ __constant__ int change_ind1[14];
__device__ __constant__ int change1[14];
__device__ __constant__ int change_ind2[14];
__device__ __constant__ int change2[14];


__device__ float num_to_mM(int n)
{
    return n * 5.5353e-4; // n/1806.6;
}

__device__ float mM_to_num(float cc)
{
    return rintf(cc * 1806.6);
}

__device__ float compute_fp( float ca_cc)
{
    float tmp = ca_cc*3.3333333333;
    tmp *= tmp;
    return tmp/(1+tmp);
}

__device__ float compute_fn( float Cstar_cc, float ns)
{
    float tmp = Cstar_cc*5.55555555;
    tmp *= tmp*tmp;
    return ns*tmp/(1+tmp);
}

__device__ float compute_ca(int Tstar, float cstar_cc, float Vm)
{
    float I_in = Tstar*8*fmaxf(-Vm,0);
    float denom = (1060 - 120*cstar_cc + 179.0952 * expf(-39.60793*Vm));
    float numer = I_in * 690.9537 + 0.0795979 + 22*cstar_cc;

    return fmaxf(1.6e-4, numer/denom);
}

__global__ void
transduction(curandStateXORWOW_t *state, float dt, %(type)s* d_Vm,
             %(type)s* g_ns, %(type)s* input,
             int* num_microvilli, int total_microvilli, int* count)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid %% 32;
    int wrp = tid >> 5;

    __shared__ int X[BLOCK_SIZE][7];  // number of molecules
    __shared__ float Ca[BLOCK_SIZE];
    __shared__ float fn[BLOCK_SIZE];

    float Vm, ns, lambda;

    float sumrate, dt_advanced;
    int reaction_ind;
    ushort2 tmp;

    // copy random generator state locally to avoid accessing global memory
    curandStateXORWOW_t localstate = state[gid];


    int mid; // microvilli ID
    volatile __shared__ int mi[4]; // starting point of mid per ward, blocksize must be 128

    // use atomicAdd to obtain the starting mid for the warp
    if(wid == 0)
    {
        mi[wrp] = atomicAdd(count, 32);
    }
    mid = mi[wrp] + wid;
    int ind;

    while(mid < total_microvilli)
    {
        ind = ((ushort*)d_X[4])[mid];
        // load variables that are needed for computing calcium concentration
        tmp = ((ushort2*)d_X[2])[mid];
        X[tid][5] = tmp.x;
        X[tid][6] = tmp.y;

        Vm = d_Vm[ind]*1e-3;
        ns = g_ns[ind];

        // update calcium concentration
        Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
        fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns);

        lambda = input[ind]/(double)num_microvilli[ind];

        // load the rest of variables
        tmp = ((ushort2*)d_X[1])[mid];
        X[tid][4] = tmp.y;
        X[tid][3] = tmp.x;
        tmp = ((ushort2*)d_X[0])[mid];
        X[tid][2] = tmp.y;
        X[tid][1] = tmp.x;
        X[tid][0] = ((ushort*)d_X[3])[mid];

        sumrate = lambda + 54198 * Ca[tid] * (0.5 - X[tid][5] * 5.5353e-4) + 5.5 * X[tid][5]; // 11, 12
        sumrate += 25 * (1+10*fn[tid]) * X[tid][6]; // 10
        sumrate += 4 * (1+37.8*fn[tid]) * X[tid][4] ; // 8
        sumrate += (1444+1598.4*fn[tid]) * X[tid][3] ; // 7, 6
        sumrate += (3.7*(1+40*fn[tid]) + 7.05 * X[tid][1]) * X[tid][0] ; // 1, 2
        sumrate += (1560 - 12.6 * X[tid][3]) * X[tid][2]; // 3, 4
        sumrate += 3.5 * (50 - X[tid][2] - X[tid][1] - X[tid][3]) ; // 5
        sumrate += 0.015 * (1+11.5*compute_fp( Ca[tid] )) * X[tid][4]*(X[tid][4]-1)*(25-X[tid][6])*0.5 ; // 9

        dt_advanced = -logf(curand_uniform(&localstate))/(LA+sumrate);

        // If the reaction time is smaller than dt,
        // pick the reaction and update,
        // then compute the total rate and next reaction time again
        // until all dt_advanced is larger than dt.
        // Note that you don't have to compensate for
        // the last reaction time that exceeds dt.
        // The reason is that the exponential distribution is MEMORYLESS.
        while (dt_advanced <= dt) {
            reaction_ind = 0;
            sumrate = curand_uniform(&localstate) * sumrate;

            if (sumrate > 2e-5) {
                sumrate -= lambda;
                reaction_ind = (sumrate<=2e-5) * 13;

                if (!reaction_ind) {
                    sumrate -= mM_to_num(30) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );
                    reaction_ind = (sumrate<=2e-5) * 11;

                    if (!reaction_ind) {
                        sumrate -= mM_to_num(5.5) * num_to_mM(X[tid][5]);
                        reaction_ind = (sumrate<=2e-5) * 12;

                        if (!reaction_ind) {
                            sumrate -= 25 * (1+10*fn[tid]) * X[tid][6];
                            reaction_ind = (sumrate<=2e-5) * 10;

                            if (!reaction_ind) {
                                sumrate -= 4 * (1+37.8*fn[tid]) * X[tid][4];
                                reaction_ind = (sumrate<=2e-5) * 8;

                                if (!reaction_ind) {
                                    sumrate -= 144 * (1+11.1*fn[tid]) * X[tid][3];
                                    reaction_ind = (sumrate<=2e-5) * 7;

                                    if (!reaction_ind) {
                                        sumrate -= 3.7*(1+40*fn[tid]) * X[tid][0];
                                        reaction_ind = (sumrate<=2e-5) * 1;

                                        if (!reaction_ind) {
                                            sumrate -= 1300 * X[tid][3];
                                            reaction_ind = (sumrate<=2e-5) * 6;

                                            if (!reaction_ind) {
                                                sumrate -= 3.0 * X[tid][2] * X[tid][3];
                                                reaction_ind = (sumrate<=2e-5) * 4;

                                                if (!reaction_ind) {
                                                    sumrate -= 15.6 * X[tid][2]
                                                        * (100-X[tid][3]);
                                                    reaction_ind = (sumrate<=2e-5) * 3;

                                                    if (!reaction_ind) {
                                                        sumrate -= 3.5 * (50 - X[tid][2]
                                                            - X[tid][1] - X[tid][3]);
                                                        reaction_ind = (sumrate<=2e-5) * 5;

                                                        if(!reaction_ind) {
                                                            sumrate -= 7.05 * X[tid][1]
                                                                * X[tid][0];
                                                            reaction_ind = (sumrate<=2e-5)
                                                                * 2;

                                                            if(!reaction_ind) {
                                                                sumrate -= 0.015 *
                                                                    (1+11.5*compute_fp( Ca[tid] )) * X[tid][4]*(X[tid][4]-1)*(25-X[tid][6])*0.5;
                                                                reaction_ind = (sumrate<=2e-5) * 9;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //int ind;

            // only up to two state variables are needed to be updated
            // update the first one.
            ind = change_ind1[reaction_ind];
            X[tid][ind] += change1[reaction_ind];

            //update the second one
            ind = change_ind2[reaction_ind];
            if (ind != 0)
                X[tid][ind] += change2[reaction_ind];

            // compute the advance time again
            Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
            fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns );

            sumrate = lambda + 54198*Ca[tid]*(0.5 - X[tid][5]*5.5353e-4)
                + 5.5*X[tid][5]; // 11, 12
            sumrate += 25*(1 + 10*fn[tid])*X[tid][6]; // 10
            sumrate += 4*(1 + 37.8*fn[tid])*X[tid][4]; // 8
            sumrate += (1444 + 1598.4*fn[tid])*X[tid][3]; // 7, 6
            sumrate += (3.7*(1 + 40*fn[tid]) + 7.05*X[tid][1])*X[tid][0]; // 1, 2
            sumrate += (1560 - 12.6*X[tid][3])*X[tid][2]; // 3, 4
            sumrate += 3.5*(50 - X[tid][2] - X[tid][1] - X[tid][3]); // 5
            sumrate += 0.015*(1 + 11.5*compute_fp( Ca[tid] ))
                *X[tid][4]*(X[tid][4] - 1)*(25 - X[tid][6])*0.5; // 9

            dt_advanced -= logf(curand_uniform(&localstate))/(LA+sumrate);

        } // end while

        ((ushort*)d_X[3])[mid] = X[tid][0];
        ((ushort2*)d_X[0])[mid] = make_ushort2(X[tid][1], X[tid][2]);
        ((ushort2*)d_X[1])[mid] = make_ushort2(X[tid][3], X[tid][4]);
        ((ushort2*)d_X[2])[mid] = make_ushort2(X[tid][5], X[tid][6]);

        if(wid == 0)
        {
            mi[wrp] = atomicAdd(count, 32);
        }
        mid = mi[wrp] + wid;
    }
    // copy the updated random generator state back to global memory
    state[gid] = localstate;
}

}
"""
    try:
        co = [compile_options[0]+' --maxrregcount=54']
    except IndexError:
        co = ['--maxrregcount=54']

    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(
        template_run % {
            "type": dtype_to_ctype(dtype),
            "block_size": block_size,
            "fletter": 'f' if scalartype == np.float32 else ''
        },
        options = co,
        no_extern_c = True)
    func = mod.get_function('transduction')
    d_X_address, d_X_nbytes = mod.get_global("d_X")
    cuda.memcpy_htod(d_X_address, Xaddress)
    d_change_ind1_address, d_change_ind1_nbytes = mod.get_global("change_ind1")
    d_change_ind2_address, d_change_ind2_nbytes = mod.get_global("change_ind2")
    d_change1_address, d_change1_nbytes = mod.get_global("change1")
    d_change2_address, d_change2_nbytes = mod.get_global("change2")
    cuda.memcpy_htod(d_change_ind1_address, change_ind1)
    cuda.memcpy_htod(d_change_ind2_address, change_ind2)
    cuda.memcpy_htod(d_change1_address, change1)
    cuda.memcpy_htod(d_change2_address, change2)

    func.prepare('PfPPPPiP')
    func.set_cache_config(cuda.func_cache.PREFER_SHARED)
    return func


def get_hh_func(dtype, compile_options):
    template = """
#define E_K (-85)
#define E_Cl (-30)
#define G_s 1.6
#define G_dr 3.5
#define G_Cl 0.006
#define G_K 0.082
#define G_nov 3.0
#define C 4


__global__ void
hh(%(type)s* I_all, %(type)s* d_V, %(type)s* d_sa, %(type)s* d_si,
   %(type)s* d_dra, %(type)s* d_dri, %(type)s* d_nov, int num_neurons,
   %(type)s ddt, int multiple)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_neurons) {
        %(type)s I = I_all[tid];
        %(type)s V = d_V[tid];  //mV
        %(type)s sa = d_sa[tid];
        %(type)s si = d_si[tid];
        %(type)s dra = d_dra[tid];
        %(type)s dri = d_dri[tid];
        %(type)s nov = d_nov[tid];

        %(type)s x_inf, tau_x, dx;
        %(type)s dt = 1000 * ddt;

        for(int i = 0; i < multiple; ++i) {
            /* The precision of power constant affects the result */
            x_inf = cbrt%(fletter)s(1/(1+exp%(fletter)s((-23.7-V)/12.8)));
            tau_x = 0.13+3.39*exp%(fletter)s(-(-73-V)*(-73-V)/400);
            dx = (x_inf - sa)/tau_x;
            sa += dt * dx;

            x_inf = 0.9/(1+exp%(fletter)s((-55-V)/-3.9))
                    + 0.1/(1+exp%(fletter)s( (-74.8-V)/-10.7));
            tau_x = 113*exp%(fletter)s(-(-71-V)*(-71-V)/841);
            dx = (x_inf - si)/tau_x;
            si += dt * dx;

            x_inf = sqrt%(fletter)s(1/(1+exp%(fletter)s((-1-V)/9.1)));
            tau_x = 0.5+5.75*exp%(fletter)s(-(-25-V)*(-25-V)/1024);
            dx = (x_inf - dra)/tau_x;
            dra += dt * dx;

            x_inf = 1/(1+exp%(fletter)s((-25.7-V)/-6.4));
            tau_x = 890;
            dx = (x_inf - dri)/tau_x;
            dri += dt * dx;

            x_inf = 1/(1+exp%(fletter)s((-12-V)/11));
            tau_x = 3 + 166*exp%(fletter)s(-(-20-V)*(-20-V)/484);
            dx = (x_inf - nov)/tau_x;
            nov += dt * dx;

            dx = (I - G_K*(V-E_K) - G_Cl * (V-E_Cl) -
                  G_s * sa*sa*sa * si * (V-E_K) -
                  G_dr * dra*dra * dri * (V-E_K)
                  - G_nov * nov * (V-E_K) )/C;
            V += dt * dx;
        }
        d_V[tid] = V;
        d_sa[tid] = sa;
        d_si[tid] = si;
        d_dra[tid] = dra;
        d_dri[tid] = dri;
        d_nov[tid] = nov;
    }
}
"""
    # Used 53 registers, 388 bytes cmem[0], 304 bytes cmem[2]
    # float: Used 35 registers, 380 bytes cmem[0], 96 bytes cmem[2]
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype), "fletter": 'f'
                                   if scalartype == np.float32 else ''},
                       options = compile_options)
    func = mod.get_function('hh')
    func.prepare('PPPPPPPi'+np.dtype(dtype).char+'i')
    return func


def get_sum_current_func(dtype, block_size, compile_options):
    template = """
#define BLOCK_SIZE %(block_size)d
#define G_TRP           8       /* conductance of a TRP channel */
#define TRP_REV 0  /* mV */

__inline__ __device__
int warpReduction(volatile int* sdata, int tid){
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    return sdata[tid] + sdata[tid+1];
}

__global__ void
sum_current(ushort2* d_Tstar, int* d_num_microvilli,
            int* d_cum_microvilli,
            %(type)s* d_Vm, %(type)s* I_all,
            %(type)s* I_fb)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int num_microvilli = d_num_microvilli[bid];
    int shift = d_cum_microvilli[bid];

    int total_open_channel;
    __shared__ int sum[BLOCK_SIZE];
    sum[tid] = 0;

    for(int i = tid; i < num_microvilli; i += BLOCK_SIZE)
        sum[tid] += d_Tstar[i + shift].y;

    __syncthreads();

    if (tid < 64) {
        #pragma unroll
        for(int i = 1; i < BLOCK_SIZE/64; ++i)
            sum[tid] += sum[tid + 64*i];
    }
    __syncthreads();

    if (tid < 32) total_open_channel = warpReduction(sum, tid);

    if (tid == 0) {
        %(type)s Vm = (d_Vm[bid]-TRP_REV) * 0.001;
        %(type)s I_in;
        if(Vm < 0)
            I_in = total_open_channel * G_TRP * (-Vm);
        else
            I_in = 0;

        I_all[bid] = I_fb[bid] + I_in / 15.7; // convert pA into \muA/cm^2
    }
}
"""
    assert(block_size%64 == 0)
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "block_size": block_size},
                       options = compile_options)
    func = mod.get_function('sum_current')
    func.prepare('PPPPPP')
    return func
