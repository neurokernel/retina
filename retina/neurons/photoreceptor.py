import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
import tables

import neurokernel.LPU.utils.curand as curand

from baseneuron import BaseNeuron


class Photoreceptor(BaseNeuron):
    dtype = np.double

    def __init__(self, n_dict, V_p, input_dt, debug, LPU_id):
        '''
        V: pointer to gpu array location where output potential is stored
        '''

        self.num_neurons = len(n_dict['id'])  # NOT n_dict['num_neurons']

        # num_microvilli must be the same for every photoreceptor so the first
        # one is taken
        self.num_microvilli = int(n_dict['num_microvilli'][0])

        self.V_p = V_p  # output of hh (pointer don't use .gpudata)

        self.input_dt = input_dt
        self.run_dt = 1e-4

        self.multiple = int(self.input_dt/self.run_dt)
        assert(self.multiple * self.run_dt == self.input_dt)

        self.record_neuron = debug
        self.debug = debug
        self.LPU_id = LPU_id

        self.block_transduction = (128, 1, 1)
        self.grid_transduction = (self.num_neurons, 1)
        self.block_hh = (256, 1, 1)
        self.grid_hh = ((self.num_neurons-1)/self.block_hh[0] + 1, 1)
        self.block_state = (32, 32, 1)
        self.grid_state = ((self.num_neurons-1)/self.block_state[0] + 1, 1)

        self._initialize(n_dict)

    @classmethod
    def initneuron(cls, n_dict, neuronstate_p, dt, debug=False, LPU_id=None):
        return cls(n_dict, neuronstate_p, dt, debug, LPU_id)

    def _initialize(self, n_dict):
        self._setup_output()
        self._setup_poisson()
        self._setup_transduction()
        self._setup_hh()
        self._setup_update_state(n_dict)

    def _setup_output(self):
        outputfile = self.LPU_id + '_out'
        if self.record_neuron:
            self.outputfile_I = tables.openFile(outputfile+'I.h5', 'w')
            self.outputfile_I.createEArray(
                "/", "array",
                tables.Float64Atom() if self.dtype == np.double else tables.Float32Atom(),
                (0, self.num_neurons))

    def _setup_poisson(self, seed=0):
        self.randState = curand.curand_setup(
            self.block_transduction[0]*self.num_neurons, seed)
        self.photon_absorption_func = get_photon_absorption_func(self.dtype)

    def _setup_transduction(self):

        self.X = []
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        tmp[:, ::2] = 50
        # variables G, Gstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        # variables PLCstar, Dstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        # variables Cstar, Tstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli), np.int16)
        # variables Mstar
        self.X.append(garray.to_gpu(tmp))

        Xaddress = np.empty(4, np.int64)
        for i in range(4):
            Xaddress[i] = int(self.X[i].gpudata)

        change_ind1 = np.asarray([1, 1, 2, 3, 3, 2, 5, 4, 5, 5, 7, 6, 6, 1],
                                 np.int32) - 1
        change_ind2 = np.asarray([1, 1, 3, 4, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1],
                                 np.int32) - 1
        change1 = np.asarray([0, -1, -1, -1, -1, 1, 1, -1, -1, -2, -1, 1, -1, 1],
                             np.int32)
        change2 = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             np.int32)

        self.transduction_func = get_transduction_func(
            self.dtype, self.block_transduction[0],
            self.num_microvilli, Xaddress,
            change_ind1, change_ind2,
            change1, change2)

        self.ns = garray.zeros(self.num_neurons, self.dtype) + 1
        self.update_ns_func = get_update_ns_func(self.dtype)

    def _setup_hh(self):
        self.I = garray.zeros((1, self.num_neurons), self.dtype)
        self.hhx = [garray.empty((1, self.num_neurons), self.dtype)
                    for _ in range(5)]

        # mV
        V_init = np.empty((1, self.num_neurons), dtype=np.double)
        V_init.fill(-0.0819925*1000)
        cuda.memcpy_htod(int(self.V_p), V_init)

        self.hhx[0].fill(0.2184)
        self.hhx[1].fill(0.9653)
        self.hhx[2].fill(0.0117)
        self.hhx[3].fill(0.9998)
        self.hhx[4].fill(0.0017)

        self.sum_current_func = get_sum_current_func(self.dtype,
                                                     self.block_transduction[0])
        self.hh_func = get_hh_func(self.dtype)

    def _setup_update_state(self, n_dict):
        self.I_fb = garray.zeros(self.num_neurons, self.dtype)
        self.photons = garray.zeros(self.num_neurons, self.dtype)

        num_dendrite_g = np.asarray([n_dict['num_dendrites_g'][i]
                                     for i in range(self.num_neurons)],
                                    dtype=np.int32).flatten()
        # Anything associated with I is actually the input which is photons/s
        num_dendrite_p = np.asarray([n_dict['num_dendrites_I'][i]
                                     for i in range(self.num_neurons)],
                                    dtype=np.int32).flatten()

        self.cumsum_dendrite_g = garray.to_gpu(np.concatenate((
                                np.asarray([0,], dtype=np.int32),
                                np.cumsum(num_dendrite_g, dtype=np.int32))))
        self.cumsum_dendrite_p = garray.to_gpu(np.concatenate((
                                np.asarray([0,], dtype=np.int32),
                                np.cumsum(num_dendrite_p, dtype=np.int32))))

        self.num_dendrite_g = garray.to_gpu(num_dendrite_g)
        self.num_dendrite_p = garray.to_gpu(num_dendrite_p)
        if len(n_dict['g_pre']):
            self.g_pre = garray.to_gpu(np.asarray(n_dict['g_pre'], dtype=np.int32))
            self.V_rev = garray.to_gpu(np.asarray(n_dict['V_rev'], dtype=np.double))
            self.fb = True
        else:
            self.fb = False

        if len(n_dict['I_pre']):
            self.p_pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
            self.get_in = True
        else:
            self.get_in = False

        self.update_I_fb = self.get_update_from_g_func(self.num_neurons)
        self.update_photons = self.get_update_from_I_func(self.num_neurons)

    def post_run(self):
        if self.record_neuron:
            self.outputfile_I.close()

    def _write_outputfile(self):
        if self.record_neuron:
            self.outputfile_I.root.array.append(self.I.get())
            self.outputfile_I.flush()


    def eval(self, st=None):
        # photons -> Metarhodopsin(X[3])
        for _ in range(self.multiple):
            if self.debug:
                minimum = min(self.photons.get())
                if (minimum < 0):
                    raise ValueError('Inputs to photoreceptor should not '
                                     'be negative, minimum value detected: {}'
                                     .format(minimum))
            # X, V, ns -> X
            self.transduction_func.prepared_call(
                self.grid_transduction, self.block_transduction,
                self.randState.gpudata, self.num_neurons, self.run_dt,
                self.V_p, self.ns.gpudata, self.photons.gpudata)

            # X, V, I_fb -> I
            self.sum_current_func.prepared_call(
                self.grid_transduction, self.block_transduction,
                self.X[2].gpudata, self.num_neurons, self.num_microvilli,
                self.V_p, self.I_fb.gpudata, self.I.gpudata)

            # hhX, I -> hhX, V
            self.hh_func.prepared_call(
                self.grid_hh, self.block_hh,
                self.I.gpudata, self.V_p, self.hhx[0].gpudata,
                self.hhx[1].gpudata, self.hhx[2].gpudata, self.hhx[3].gpudata,
                self.hhx[4].gpudata, self.num_neurons, self.run_dt/10, 10)

        self.update_ns_func.prepared_call(
            ( (self.num_neurons - 1) / 128 + 1, 1), (128, 1, 1),
            self.ns.gpudata, self.num_neurons, self.V_p, self.run_dt)

        self._write_outputfile()

    def update_internal_state(self, synapse_state_p, st=None, logger=None):
        # feedback current
        if self.fb:
            self.I_fb.fill(0)
            self.update_I_fb.prepared_async_call(
                self.grid_state, self.block_state, st,
                int(synapse_state_p), self.cumsum_dendrite_g.gpudata,
                self.num_dendrite_g.gpudata, self.g_pre.gpudata,
                self.I_fb.gpudata, int(self.V_p), self.V_rev.gpudata)

        if self.get_in:
            self.photons.fill(0)
            self.update_photons.prepared_async_call(
                self.grid_state, self.block_state, st,
                int(synapse_state_p), self.cumsum_dendrite_p.gpudata,
                self.num_dendrite_p.gpudata, self.p_pre.gpudata,
                self.photons.gpudata)

# end of photoreceptor


def get_photon_absorption_func(dtype):
    template = """

#include "curand_kernel.h"
extern "C" {
__global__ void
photon_absorption(curandStateXORWOW_t *state, short* M, int num_neurons,
                  int num_microvilli, %(type)s* input)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;

    // XXX What happens if bid is not within array limits?
    %(type)s lambda = input[bid] / num_microvilli;

    int n_photon;

    curandStateXORWOW_t localstate = state[bdim*bid + tid];

    // each thread will update the values of a few microvilli
    // of a certain neuron with id 'bid'
    for(int i = tid; i < num_microvilli; i += bdim)
    {
        n_photon = curand_poisson(&localstate, lambda);
        if(n_photon)
        {
            M[i + bid * num_neurons] += n_photon;
        }
    }
    state[bdim*bid + tid] = localstate;
}

}
"""
# Used 33 registers, 352 bytes cmem[0], 328 bytes cmem[2]
# float: Used 47 registers, 352 bytes cmem[0], 332 bytes cmem[2]
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"],
                       no_extern_c = True)
    func = mod.get_function('photon_absorption')
    func.prepare([np.intp, np.intp, np.int32, np.int32, np.intp])
    return func


def get_update_ns_func(dtype):
    template = """

#define RTAU 1.0
__global__ void
update_ns(%(type)s* g_ns, int num_neurons, %(type)s* V, %(type)s dt)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < num_neurons)
    {
        %(type)s v = V[tid];
        %(type)s ns = g_ns[tid];
        %(type)s n_inf;

        if(v >= -53)
        {
            n_inf = 8.5652*(v+53)+5;
        } else
        {
            n_inf = fmax%(fletter)s(1, 0.2354*(v+70)+1);
        }

        g_ns[tid] = ns + (n_inf-ns)*RTAU*dt;

    }
}
"""
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "fletter": 'f' if scalartype == np.float32 else ''},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('update_ns')
    func.prepare([np.intp, np.int32, np.intp, scalartype])
    return func


def get_transduction_func(dtype, block_size, num_microvilli, Xaddress,
                          change_ind1, change_ind2, change1, change2):
    template = """

#include "curand_kernel.h"

extern "C" {
#include "stdio.h"

#define NUM_MICROVILLI %(num_microvilli)d
#define BLOCK_SIZE %(block_size)d
#define LA 0.5

/* Simulation Constants */
#define C_T     0.5     /* Total concentration of calmodulin */
#define G_T     50      /* Total number of G-protein */
#define PLC_T   100     /* Total number of PLC */
#define T_T     25      /* Total number of TRP/TRPL channels */
#define I_TSTAR 0.68    /* Average current through one opened TRP/TRPL channel (pA)*/

#define GAMMA_DSTAR     4.0 /* s^(-1) rate constant*/
#define GAMMA_GAP       3.0 /* s^(-1) rate constant*/
#define GAMMA_GSTAR     3.5 /* s^(-1) rate constant*/
#define GAMMA_MSTAR     3.7 /* s^(-1) rate constant*/
#define GAMMA_PLCSTAR   144 /* s^(-1) rate constant */
#define GAMMA_TSTAR     25  /* s^(-1) rate constant */

#define H_DSTAR         37.8    /* strength constant */
#define H_MSTAR         40      /* strength constant */
#define H_PLCSTAR       11.1    /* strength constant */
#define H_TSTARP        11.5    /* strength constant */
#define H_TSTARN        10      /* strength constant */

#define K_P     0.3     /* Dissociation coefficient for calcium positive feedback */
#define K_P_INV 3.3333  /* K_P inverse ( too many decimals are not important) */
#define K_N     0.18    /* Dissociation coefficient for calmodulin negative feedback */
#define K_N_INV 5.5555  /* K_N inverse ( too many decimals are not important) */
#define K_U     30      /* (mM^(-1)s^(-1)) Rate of Ca2+ uptake by calmodulin */
#define K_R     5.5     /* (mM^(-1)s^(-1)) Rate of Ca2+ release by calmodulin */
#define K_CA    1000    /* s^(-1) diffusion from microvillus to somata (tuned) */

#define K_NACA  3e-8    /* Scaling factor for Na+/Ca2+ exchanger model */

#define KAPPA_DSTAR         1300.0  /* s^(-1) rate constant - there is also a capital K_DSTAR */
#define KAPPA_GSTAR         7.05    /* s^(-1) rate constant */
#define KAPPA_PLCSTAR       15.6    /* s^(-1) rate constant */
#define KAPPA_TSTAR         150.0   /* s^(-1) rate constant */
#define K_DSTAR             100.0   /* rate constant */

#define F                   96485   /* (mC/mol) Faraday constant (changed from paper)*/
#define N                   4       /* Binding sites for calcium on calmodulin */
#define R                   8.314   /* (J*K^-1*mol^-1)Gas constant */
#define T                   293     /* (K) Absolute temperature */
#define VOL                 3e-9    /* changed from 3e-12microlitres to nlitres
                                     * microvillus volume so that units agree */

#define N_S0_DIM        1   /* initial condition */
#define N_S0_BRIGHT     2

#define A_N_S0_DIM      4   /* upper bound for dynamic increase (of negetive feedback) */
#define A_N_S0_BRIGHT   200

#define TAU_N_S0_DIM    3000    /* time constant for negative feedback */
#define TAU_N_S0_BRIGHT 1000

#define NA_CO           120     /* (mM) Extracellular sodium concentration */
#define NA_CI           8       /* (mM) Intracellular sodium concentration */
#define CA_CO           1.5     /* (mM) Extracellular calcium concentration */

#define G_TRP           8       /* conductance of a TRP channel */
#define TRP_REV         0       /* TRP channel reversal potential (mV) */

__device__ __constant__ long long int d_X[4];
__device__ __constant__ int change_ind1[13];
__device__ __constant__ int change1[13];
__device__ __constant__ int change_ind2[13];
__device__ __constant__ int change2[13];

/* cc = n/(NA*VOL) [6.0221413e+23 mol^-1 * 3*10e-21 m^3] */
__device__ float num_to_mM(int n)
{
    return n * 5.5353e-4; // n/1806.6;
}

/* n = cc*VOL*NA [6.0221413e+23 mol^-1 * 3*10e-21 m^3] */
__device__ float mM_to_num(float cc)
{
    return rintf(cc * 1806.6);
}

/* Assumes Hill constant (=2) for positive calcium feedback */
__device__ float compute_fp(float Ca_cc)
{
    float tmp = Ca_cc*K_P_INV;
    tmp *= tmp;
    return tmp/(1 + tmp);
}

/* Assumes Hill constant(=3) for negative calmodulin feedback */
__device__ float compute_fn(float Cstar_cc, float ns)
{
    float tmp = Cstar_cc*K_N_INV;
    tmp *= tmp*tmp;
    return ns*tmp/(1 + tmp);
}

/* Vm [V] */
__device__ float compute_ca(int Tstar, float Cstar_cc, float Vm)
{
    float I_in = Tstar*G_TRP*fmaxf(-Vm + 0.001*TRP_REV, 0);
    /* CaM = C_T - Cstar_cc */
    float denom = (K_CA + (N*K_U*C_T) - (N*K_U)*Cstar_cc + 179.0952 * expf(-(F/(R*T))*Vm));  // (K_NACA*NA_CO^3/VOL*F)
    /* I_Ca ~= 0.4*I_in */
    float numer = (0.4*I_in)/(2*VOL*F) +
                  ((K_NACA*CA_CO*NA_CI*NA_CI*NA_CI)/(VOL*F)) +  // in paper it's -K_NACA... due to different conventions
                  N*K_R*Cstar_cc;

    return fmaxf(1.6e-4, numer/denom);
}

__global__ void
transduction(curandStateXORWOW_t *state, int num_neurons,
             float dt, %(type)s* d_Vm, %(type)s* g_ns, %(type)s* input)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int X[BLOCK_SIZE][7];  // number of molecules
    __shared__ float Ca[BLOCK_SIZE];
    __shared__ float Vm;  // membrane voltage, shared over all threads
    __shared__ float ns;
    __shared__ float fn[BLOCK_SIZE];
    __shared__ float lambda;

    if(tid == 0)
    {
        Vm = d_Vm[bid] * 0.001;  // V
        ns = g_ns[bid];
        lambda = input[bid]/NUM_MICROVILLI; //input must have unit photons/sec
    }

    __syncthreads();


    float sumrate;
    float dt_advanced;
    int reaction_ind;
    short2 tmp;

    // copy random generator state locally to avoid accessing global memory
    curandStateXORWOW_t localstate = state[BLOCK_SIZE*bid + tid];

    // iterate over all microvilli in one photoreceptor
    for(int i = tid; i < NUM_MICROVILLI; i += BLOCK_SIZE)
    {
        // load variables that are needed for computing calcium concentration
        //Ca[tid] = ((%(type)s*)d_X[7])[bid*ld2 + i]; // no need to store calcium
        tmp = ((short2*)d_X[2])[bid*num_neurons + i];
        X[tid][5] = tmp.x;
        X[tid][6] = tmp.y;

        // update calcium concentration
        Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
        fn[tid] = compute_fn(num_to_mM(X[tid][5]), ns);

        // load the rest of variables
        tmp = ((short2*)d_X[1])[bid*num_neurons + i];
        X[tid][4] = tmp.y;
        X[tid][3] = tmp.x;
        tmp = ((short2*)d_X[0])[bid*num_neurons + i];
        X[tid][2] = tmp.y;
        X[tid][1] = tmp.x;
        X[tid][0] = ((short*)d_X[3])[bid*num_neurons + i];

        // compute total rate of reaction
        sumrate = lambda;
        sumrate += mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );  //11
        sumrate += mM_to_num(K_R) * num_to_mM(X[tid][5]);  //12
        sumrate += GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6];  // 10
        sumrate += GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4];  // 8
        sumrate += GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3];  // 7
        sumrate += GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0];  // 1
        sumrate += KAPPA_DSTAR * X[tid][3];  // 6
        sumrate += GAMMA_GAP * X[tid][2] * X[tid][3];  // 4
        sumrate += KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);  // 3
        sumrate += GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]);  // 5
        sumrate += KAPPA_GSTAR * X[tid][1] * X[tid][0];  // 2
        sumrate += (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                   (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                   X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5 ;  // 9

        // choose the next reaction time
        dt_advanced = -logf(curand_uniform(&localstate))/(LA + sumrate);

        // If the reaction time is smaller than dt,
        // pick the reaction and update,
        // then compute the total rate and next reaction time again
        // until all dt_advanced is larger than dt.
        // Note that you don't have to compensate for
        // the last reaction time that exceeds dt.
        // The reason is that the exponential distribution is MEMORYLESS.
        while(dt_advanced <= dt)
        {
            reaction_ind = 0;
            sumrate = curand_uniform(&localstate) * sumrate;

            if(sumrate > 2e-5)
            {

                sumrate -= lambda;
                reaction_ind = (sumrate<=2e-5) * 13;

                if(!reaction_ind)
                {

                    sumrate -= mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );
                    reaction_ind = (sumrate<=2e-5) * 11;

                    if(!reaction_ind)
                    {
                        sumrate -= mM_to_num(K_R) * num_to_mM(X[tid][5]);
                        reaction_ind = (sumrate<=2e-5) * 12;
                        if(!reaction_ind)
                        {
                            sumrate -= GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6];
                            reaction_ind = (sumrate<=2e-5) * 10;
                            if(!reaction_ind)
                            {
                                sumrate -= GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4];
                                reaction_ind = (sumrate<=2e-5) * 8;

                                if(!reaction_ind)
                                {
                                    sumrate -= GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3];
                                    reaction_ind = (sumrate<=2e-5) * 7;
                                    if(!reaction_ind)
                                    {
                                        sumrate -= GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0];
                                        reaction_ind = (sumrate<=2e-5) * 1;
                                        if(!reaction_ind)
                                        {
                                            sumrate -= KAPPA_DSTAR * X[tid][3];
                                            reaction_ind = (sumrate<=2e-5) * 6;
                                            if(!reaction_ind)
                                            {
                                                sumrate -= GAMMA_GAP * X[tid][2] * X[tid][3];
                                                reaction_ind = (sumrate<=2e-5) * 4;

                                                if(!reaction_ind)
                                                {
                                                    sumrate -= KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);
                                                    reaction_ind = (sumrate<=2e-5) * 3;
                                                    if(!reaction_ind)
                                                    {
                                                        sumrate -= GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]);
                                                        reaction_ind = (sumrate<=2e-5) * 5;
                                                        if(!reaction_ind)
                                                        {
                                                            sumrate -= KAPPA_GSTAR * X[tid][1] * X[tid][0];
                                                            reaction_ind = (sumrate<=2e-5) * 2;
                                                            if(!reaction_ind)
                                                            {
                                                                sumrate -= (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                                                                           (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                                                                           X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5;
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
            int ind;

            // only up to two state variables are needed to be updated
            // update the first one.
            ind = change_ind1[reaction_ind];
            X[tid][ind] += change1[reaction_ind];

            //if(reaction_ind == 9)
            //{
            //    X[tid][ind] = max(X[tid][ind], 0);
            //}

            ind = change_ind2[reaction_ind];
            //update the second one
            if(ind != 0)
            {
                X[tid][ind] += change2[reaction_ind];
            }

            // compute the advance time again
            Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
            fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns );
            //fp[tid] = compute_fp( Ca[tid] );

            sumrate = lambda;
            sumrate += mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) ); //11
            sumrate += mM_to_num(K_R) * num_to_mM(X[tid][5]); //12
            sumrate += GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6]; // 10
            sumrate += GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4]; // 8
            sumrate += GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3]; // 7
            sumrate += GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0]; // 1
            sumrate += KAPPA_DSTAR * X[tid][3]; // 6
            sumrate += GAMMA_GAP * X[tid][2] * X[tid][3]; // 4
            sumrate += KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);  // 3
            sumrate += GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]); // 5
            sumrate += KAPPA_GSTAR * X[tid][1] * X[tid][0]; // 2
            sumrate += (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                       (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                       X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5; // 9

            dt_advanced -= logf(curand_uniform(&localstate))/(LA + sumrate);

        } // end while

        ((short*)d_X[3])[bid*num_neurons + i] = X[tid][0];
        ((short2*)d_X[0])[bid*num_neurons + i] = make_short2(X[tid][1], X[tid][2]);
        ((short2*)d_X[1])[bid*num_neurons + i] = make_short2(X[tid][3], X[tid][4]);
        ((short2*)d_X[2])[bid*num_neurons + i] = make_short2(X[tid][5], X[tid][6]);
    }
    // copy the updated random generator state back to global memory
    state[BLOCK_SIZE*bid + tid] = localstate;
}

}
"""
    #ptxas info    : 77696 bytes gmem, 336 bytes cmem[3]
    #ptxas info    : Compiling entry function 'transduction' for 'sm_35'
    #ptxas info    : Function properties for transduction
    #    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    #ptxas info    : Used 60 registers, 7176 bytes smem, 352 bytes cmem[0], 324 bytes cmem[2]
    #float : Used 65 registers, 7172 bytes smem, 344 bytes cmem[0], 168 bytes cmem[2]

    template_run = """

#include "curand_kernel.h"

extern "C" {
#include "stdio.h"

#define NUM_MICROVILLI %(num_microvilli)d
#define BLOCK_SIZE %(block_size)d
#define LA 0.5

__device__ __constant__ long long int d_X[4];
__device__ __constant__ int change_ind1[13];
__device__ __constant__ int change1[13];
__device__ __constant__ int change_ind2[13];
__device__ __constant__ int change2[13];


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
transduction(curandStateXORWOW_t *state, int ld1,
             float dt, %(type)s* d_Vm, %(type)s* g_ns, %(type)s* input)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int X[BLOCK_SIZE][7]; // number of molecules
    __shared__ float Ca[BLOCK_SIZE];
    __shared__ float Vm[1]; // membrane voltage, shared over all threads
    __shared__ float ns;
    __shared__ float fn[BLOCK_SIZE];
    __shared__ float lambda;

    if(tid == 0)
    {
        Vm[0] = d_Vm[bid]*1e-3;
        ns = g_ns[bid];
        lambda = input[bid]/NUM_MICROVILLI; //input must have unit photons/sec
    }

    __syncthreads();

    float sumrate;
    float rnumber;
    float dt_advanced;
    int reaction_ind;
    short2 tmp;

    // copy random generator state locally to avoid accessing global memory
    curandStateXORWOW_t localstate = state[BLOCK_SIZE*bid + tid];

    // iterate over all microvilli in one photoreceptor
    for(int i = tid; i < NUM_MICROVILLI; i += BLOCK_SIZE)
    {
        // load variables that are needed for computing calcium concentration
        tmp = ((short2*)d_X[2])[bid*ld1 + i];
        X[tid][5] = tmp.x;
        X[tid][6] = tmp.y;

        // update calcium concentration
        Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm[0]);
        fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns);

        // load the rest of variables
        tmp = ((short2*)d_X[1])[bid*ld1 + i];
        X[tid][4] = tmp.y;
        X[tid][3] = tmp.x;
        tmp = ((short2*)d_X[0])[bid*ld1 + i];
        X[tid][2] = tmp.y;
        X[tid][1] = tmp.x;
        X[tid][0] = ((short*)d_X[3])[bid*ld1 + i];

        sumrate = lambda + 54198 * Ca[tid] * (0.5 - X[tid][5] * 5.5353e-4) + 5.5 * X[tid][5]; // 11, 12
        sumrate += 25 * (1+10*fn[tid]) * X[tid][6]; // 10
        sumrate += 4 * (1+37.8*fn[tid]) * X[tid][4] ; // 8
        sumrate += (1444+1598.4*fn[tid]) * X[tid][3] ; // 7, 6
        sumrate += (3.7*(1+40*fn[tid]) + 7.05 * X[tid][1]) * X[tid][0] ; // 1, 2
        sumrate += (1560 - 12.6 * X[tid][3]) * X[tid][2]; // 3, 4
        sumrate += 3.5 * (50 - X[tid][2] - X[tid][1] - X[tid][3]) ; // 5
        sumrate += 0.015 * (1+11.5*compute_fp( Ca[tid] )) * X[tid][4]*(X[tid][4]-1)*(25-X[tid][6])*0.5 ; // 9

        dt_advanced = -logf(curand_uniform(&localstate))/(LA+sumrate);

        int counter = 0;

        // If the reaction time is smaller than dt,
        // pick the reaction and update,
        // then compute the total rate and next reaction time again
        // until all dt_advanced is larger than dt.
        // Note that you don't have to compensate for
        // the last reaction time that exceeds dt.
        // The reason is that the exponential distribution is MEMORYLESS.
        while(dt_advanced <= dt)
        {
            reaction_ind = 0;
            sumrate = curand_uniform(&localstate) * sumrate;

            if(sumrate > 2e-5)
            {
                sumrate -= lambda;
                reaction_ind = (sumrate<=2e-5) * 13;

                if(!reaction_ind)
                {

                    sumrate -= mM_to_num(30) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );
                    reaction_ind = (sumrate<=2e-5) * 11;

                    if(!reaction_ind)
                    {
                        sumrate -= mM_to_num(5.5) * num_to_mM(X[tid][5]) ;
                        reaction_ind = (sumrate<=2e-5) * 12;
                        if(!reaction_ind)
                        {
                            sumrate -= 25 * (1+10*fn[tid]) * X[tid][6] ;
                            reaction_ind = (sumrate<=2e-5) * 10;
                            if(!reaction_ind)
                            {
                                sumrate -= 4 * (1+37.8*fn[tid]) * X[tid][4] ;
                                reaction_ind = (sumrate<=2e-5) * 8;

                                if(!reaction_ind)
                                {
                                    sumrate -= 144 * (1+11.1*fn[tid]) * X[tid][3] ;
                                    reaction_ind = (sumrate<=2e-5) * 7;
                                    if(!reaction_ind)
                                    {
                                        sumrate -= 3.7*(1+40*fn[tid]) * X[tid][0];
                                        reaction_ind = (sumrate<=2e-5) * 1;
                                        if(!reaction_ind)
                                        {
                                            sumrate -= 1300 * X[tid][3] ;
                                            reaction_ind = (sumrate<=2e-5) * 6;
                                            if(!reaction_ind)
                                            {
                                                sumrate -= 3.0 * X[tid][2] * X[tid][3] ;
                                                reaction_ind = (sumrate<=2e-5) * 4;

                                                if(!reaction_ind)
                                                {
                                                    sumrate -= 15.6 * X[tid][2] * (100-X[tid][3]) ;
                                                    reaction_ind = (sumrate<=2e-5) * 3;
                                                    if(!reaction_ind)
                                                    {
                                                        sumrate -= 3.5 * (50 - X[tid][2] - X[tid][1] - X[tid][3]) ;
                                                        reaction_ind = (sumrate<=2e-5) * 5;
                                                        if(!reaction_ind)
                                                        {
                                                            sumrate -= 7.05 * X[tid][1] * X[tid][0] ;
                                                            reaction_ind = (sumrate<=2e-5) * 2;
                                                            if(!reaction_ind)
                                                            {
                                                                sumrate -= 0.015 * (1+11.5*compute_fp( Ca[tid] )) * X[tid][4]*(X[tid][4]-1)*(25-X[tid][6])*0.5;
                                                                reaction_ind = (sumrate<=2e-5) * 9;
                                                                //if(X[tid][4] < 2)
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
            int ind;

            // only up to two state variables are needed to be updated
            // update the first one.
            ind = change_ind1[reaction_ind];
            X[tid][ind] += change1[reaction_ind];

            ind = change_ind2[reaction_ind];
            //update the second one
            if(ind != 0)
            {
                X[tid][ind] += change2[reaction_ind];
            }

            // compute the advance time again
            Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm[0]);
            fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns );
            //fp[tid] = compute_fp( Ca[tid] );


            sumrate = lambda+54198 * Ca[tid] * (0.5 - X[tid][5] * 5.5353e-4) + 5.5 * X[tid][5]; // 11, 12
            sumrate += 25 * (1+10*fn[tid]) * X[tid][6]; // 10
            sumrate += 4 * (1+37.8*fn[tid]) * X[tid][4] ; // 8
            sumrate += (1444+1598.4*fn[tid]) * X[tid][3] ; // 7, 6
            sumrate += (3.7*(1+40*fn[tid]) + 7.05 * X[tid][1]) * X[tid][0] ; // 1, 2
            sumrate += (1560 - 12.6 * X[tid][3]) * X[tid][2]; // 3, 4
            sumrate += 3.5 * (50 - X[tid][2] - X[tid][1] - X[tid][3]) ; // 5
            sumrate += 0.015 * (1+11.5*compute_fp( Ca[tid] )) * X[tid][4]*(X[tid][4]-1)*(25-X[tid][6])*0.5 ; // 9

            dt_advanced -= logf(curand_uniform(&localstate))/(LA+sumrate);

        } // end while

        ((short*)d_X[3])[bid*ld1 + i] = X[tid][0];
        ((short2*)d_X[0])[bid*ld1 + i] = make_short2(X[tid][1], X[tid][2]);
        ((short2*)d_X[1])[bid*ld1 + i] = make_short2(X[tid][3], X[tid][4]);
        ((short2*)d_X[2])[bid*ld1 + i] = make_short2(X[tid][5], X[tid][6]);
    }
    // copy the updated random generator state back to global memory
    state[BLOCK_SIZE*bid + tid] = localstate;
}

}
"""
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(
        template_run % {
            "type": dtype_to_ctype(dtype),
            "block_size": block_size,
            "num_microvilli": num_microvilli,
            "fletter": 'f' if scalartype == np.float32 else ''
        },
        options = ["--ptxas-options=-v --maxrregcount=56"],
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

    func.prepare([np.intp, np.int32, np.float32, np.intp, np.intp, np.intp])
    func.set_cache_config(cuda.func_cache.PREFER_SHARED)
    return func


def get_hh_func(dtype):
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

    if(tid < num_neurons)
    {
        %(type)s I = I_all[tid];
        %(type)s V = d_V[tid];  //mV
        %(type)s sa = d_sa[tid];
        %(type)s si = d_si[tid];
        %(type)s dra = d_dra[tid];
        %(type)s dri = d_dri[tid];
        %(type)s nov = d_nov[tid];

        %(type)s x_inf, tau_x, dx;
        %(type)s dt = 1000 * ddt;

        for(int i = 0; i < multiple; ++i)
        {
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
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('hh')
    func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp,
                  np.int32, scalartype, np.int32])
    return func


def get_sum_current_func(dtype, block_size):
    template = """
#define BLOCK_SIZE %(block_size)d
#define G_TRP           8       /* conductance of a TRP channel */
#define TRP_REV 0  /* mV */

__global__ void
sum_current(short2* d_Tstar, int num_neurons, int num_microvilli, %(type)s* d_Vm,
            %(type)s* I_fb, %(type)s* I_all)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int sum[BLOCK_SIZE];
    sum[tid] = 0;

    for(int i = tid; i < num_microvilli; i += BLOCK_SIZE)
    {
        sum[tid] += d_Tstar[i + bid*num_neurons].y;
    }
    __syncthreads();

    if(tid < 32)
    {
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE/32; ++i)
        {
            sum[tid] += sum[tid + 32*i];
        }
    }

    if(tid < 16)
    {
        sum[tid] += sum[tid+16];
    }

    if(tid < 8)
    {
        sum[tid] += sum[tid+8];
    }

    if(tid < 4)
    {
        sum[tid] += sum[tid+4];
    }

    if(tid < 2)
    {
        sum[tid] += sum[tid+2];
    }

    if(tid == 0)
    {
        // %(type)s Vm = d_Vm[bid];
        %(type)s Vm = (d_Vm[bid]-TRP_REV) * 0.001;
        %(type)s I_in;
        if(Vm < 0)
        {
            I_in = (sum[tid]+sum[tid+1]) * G_TRP * (-Vm);
        }
        else
        {
            I_in = 0;
        }

        I_all[bid] = I_fb[bid] + I_in / 15.7; // convert pA into \muA/cm^2
    }
}
"""
    # Used 18 registers, 512 bytes smem, 352 bytes cmem[0], 24 bytes cmem[2]
    # float: Used 20 registers, 1024 bytes smem, 352 bytes cmem[0], 24 bytes cmem[2]
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "block_size": block_size},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('sum_current')
    func.prepare([np.intp, np.int32, np.int32, np.intp, np.intp, np.intp])
    return func

