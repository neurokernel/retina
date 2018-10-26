from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
PI = np.pi


import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.compiler import SourceModule

from .utils import parray as parray
from .utils import linalg as la


class RF(object):

    __metaclass__ = ABCMeta

    def __init__(self, grid):
        """
            grid: a meshgrid list with arrays for 2 coordinates
        """

        if grid[0].ndim == 1:
            grid = np.meshgrid(grid[0], grid[1])
        else:
            grid = grid

        self.dtype = np.dtype(np.double)
        self.grid = grid
        self.size = grid[0].size

        # flags
        self.kernel_set = False
        self.parameter_set = False
        self.gpu_loaded = False

    @abstractmethod
    def load_kernel(self):
        pass

    @abstractmethod
    def to_gpu(self):
        pass

    def _load_gpu(self):
        if not self.parameter_set:
            print('Using default filter parameters')
            self.set_parameters()
        self.to_gpu()
        self.load_kernel()
        self.gpu_loaded = True

    @abstractmethod
    def load_parameters(self, **kwargs):
        self._load_gpu()

    @abstractproperty
    def refa(self):
        pass

    @abstractproperty
    def refb(self):
        pass

    @abstractmethod
    def _call_filter_func(self, N_filters, startbias):
        pass

    def generate_filters(self, N_filters=None, startbias=0):
        """
        Generate a batch of filters from parameters set in self

        start_bias: start from the (start_bias)th filter
        N_filters: generate N_filters filters
        """
        assert self.gpu_loaded
        if N_filters is None:
            N_filters = self.num_neurons - startbias

        if hasattr(self, 'filters'):
            if N_filters != self.filters.shape[0]:
                delattr(self, 'filters')
                self.filters = parray.empty(
                    (N_filters, self.size), self.dtype)
        else:
            self.filters = parray.empty(
                (N_filters, self.size), self.dtype)

        self._call_filter_func(N_filters, startbias)

    def filter(self, video_input):
        """
        Performs RF filtering on input video
        for all the rfs
        """
        if len(video_input.shape) == 2:
            # if input has 2 dimensions
            assert video_input.shape[1] == self.size
        else:
            # if input has 3 dimensions
            assert (video_input.shape[1]*video_input.shape[2] ==
                    self.size)
        # rasterizing inputs
        video_input.resize((video_input.shape[0], self.size))

        d_video = parray.to_gpu(video_input)
        d_output = parray.empty((self.num_neurons, video_input.shape[0]),
                                self.dtype)
        free, total = cuda.mem_get_info()
        self.ONE_TIME_FILTERS = ((free // self.dtype.itemsize)
                                 * 3 // 4 // self.size)
        self.ONE_TIME_FILTERS -= self.ONE_TIME_FILTERS % 2
        self.ONE_TIME_FILTERS = min(self.ONE_TIME_FILTERS, self.num_neurons)
        handle = la.cublashandle()

        for i in np.arange(0, self.num_neurons, self.ONE_TIME_FILTERS):
            Nfilters = min(self.ONE_TIME_FILTERS, self.num_neurons - i)
            self.generate_filters(startbias=i, N_filters=Nfilters)
            la.dot(self.filters, d_video, opb='t',
                   C=d_output[i: i+Nfilters],
                   handle=handle)
        del self.filters
        return d_output.T()

    def filter_image(self, image_input):
        """
        Performs RF filtering on input video
        for all the rfs
        """
        # video dimensions should match screen dimensions
        # numpy resize operation doesn,t make any checks
        if len(image_input.shape) == 2:
            # if input has 2 dimensions
            assert image_input.shape[1] == self.size
        else:
            # if input has 3 dimensions
            assert (image_input.shape[1]*image_input.shape[2] ==
                    self.size)

        # rasterizing inputs
        image_input.resize((1, self.size))

        d_image = parray.to_gpu(image_input)
        d_output = parray.empty((self.num_neurons, image_input.shape[0]),
                                self.dtype)
        free, total = cuda.mem_get_info()
        self.ONE_TIME_FILTERS = ((free // self.dtype.itemsize)
                                 * 3 // 4 // self.size)
        self.ONE_TIME_FILTERS -= self.ONE_TIME_FILTERS % 2
        self.ONE_TIME_FILTERS = min(self.ONE_TIME_FILTERS, self.num_neurons)
        handle = la.cublashandle()

        for i in np.arange(0, self.num_neurons, self.ONE_TIME_FILTERS):
            Nfilters = min(self.ONE_TIME_FILTERS, self.num_neurons - i)
            self.generate_filters(startbias=i, N_filters=Nfilters)
            la.dot(self.filters, d_image, opb='t',
                   C=d_output[i: i+Nfilters],
                   handle=handle)
        del self.filters
        return d_output.T()

    def filter_image_use(self, image_input):
        """
        Performs RF filtering on input video
        for all the rfs
        """
        # video dimensions should match screen dimensions
        # numpy resize operation doesn,t make any checks
        if len(image_input.shape) == 2:
            # if input has 2 dimensions
            assert image_input.shape[1] == self.size
        else:
            # if input has 3 dimensions
            assert (image_input.shape[1]*image_input.shape[2] ==
                    self.size)

        # rasterizing inputs
        image_input.resize((1, self.size))

        d_image = parray.to_gpu(image_input)
        handle = la.cublashandle()

        return la.dot(self.filters, d_image, opb='t', handle=handle).T()


class Sphere_Gaussian_RF(RF):
    def __init__(self, grid):
        super(Sphere_Gaussian_RF, self).__init__(grid)

    def load_kernel(self):
        self.filter_func = _get_von_mises_fisher_rf(self.dtype)

        self._nthreads = (128, 1, 1)
        self._nblocks = ((self.size-1) // self._nthreads[0] + 1, 1)
        self.kernel_set = True

    def to_gpu(self):
        self.d_refelev = parray.to_gpu(self.refelev)
        self.d_refazim = parray.to_gpu(self.refazim)
        self.d_grid = [parray.to_gpu(self.grid[i].reshape(-1))
                       for i in range(len(self.grid))]

    def load_parameters(self, **kwargs):
        self.refelev = kwargs.get('refa').astype(self.dtype)
        self.refazim = kwargs.get('refb').astype(self.dtype)
        self.acceptance_angle = kwargs.get('acceptance_angle')
        self.radius = kwargs.get('radius')
        M = int(kwargs.get('M', 1))

        self.num_neurons = self.refelev.size
        assert self.refazim.size == self.num_neurons

        # acceptance angle is defined as half width:
        # i.e., the distance between points that has half the maximum value.
        # von mises fisher is defined as Cexp(kappa*u^T*x)
        # maximum value is Cexp(kappa)
        # angle between x and u is u^Tx
        # Cexp(kappa)/2=Cexp(kappa*cos(acceptance_angle/2))
        # solving above equation we have
        # kappa = log(2)/(1-cos(acceptance_angle/2))
        # ! divided by M to adjust receptive angle
        self.kappa = np.log(2)/(1-np.cos(self.acceptance_angle*np.pi/180/2/M))

        self.dxy = np.diff(self.grid[0][0, :2])*np.diff(self.grid[1][:2, 0])[0]
        self.parameter_set = True

        super(Sphere_Gaussian_RF, self).load_parameters()

    @property
    def refa(self):
        try:
            return self.refelev
        except:
            print('RF parameters were not loaded correctly')
            raise

    @property
    def refb(self):
        try:
            return self.refazim
        except:
            print('RF parameters were not loaded correctly')
            raise

    def _call_filter_func(self, N_filters, startbias):
        itemsize = self.dtype.itemsize
        self.filter_func.prepared_call(
            self._nblocks,
            self._nthreads,
            self.filters.gpudata,
            self.filters.ld,
            self.d_grid[0].gpudata,
            self.d_grid[1].gpudata,
            int(self.d_refelev.gpudata) + startbias * itemsize,
            int(self.d_refazim.gpudata) + startbias * itemsize,
            self.size, N_filters,
            self.dxy, self.kappa)


class Cylinder_Gaussian_RF(RF):
    def __init__(self, grid):
        super(Cylinder_Gaussian_RF, self).__init__(grid)

    def load_kernel(self):
        self.filter_func = _get_gaussian_cylinder(self.dtype)

        self._nthreads = (128, 1, 1)
        self._nblocks = ((self.size-1) // self._nthreads[0] + 1, 1)
        self.kernel_set = True

    def to_gpu(self):
        self.d_refz = parray.to_gpu(self.refz)
        self.d_reftheta = parray.to_gpu(self.reftheta)
        self.d_grid = [parray.to_gpu(self.grid[i].reshape(-1))
                       for i in range(len(self.grid))]

    def load_gpu(self):
        super(Cylinder_Gaussian_RF, self).load_gpu()

    def load_parameters(self, **kwargs):
        self.refz = kwargs.get('refa').astype(self.dtype)
        self.reftheta = kwargs.get('refb').astype(self.dtype)
        self.acceptance_angle = kwargs.get('acceptance_angle')  # degrees
        self.radius = kwargs.get('radius')
        M = int(kwargs.get('M', 1))

        self.num_neurons = self.refz.size
        assert self.reftheta.size == self.num_neurons

        # calculations below assume radius = 1

        # acceptance angle is defined as half width:
        # i.e., the distance between points that has half the maximum value.
        # von mises fisher is defined as Cexp(kappa*u^T*x)
        # maximum value is Cexp(kappa)
        # angle between x and u is u^Tx
        # Cexp(kappa)/2=Cexp(kappa*cos(acceptance_angle/2))
        # solving above equation we have
        # kappa = log(2)/(1-cos(acceptance_angle/2))
        # ! divided by M to adjust receptive angle
        self.kappa = np.log(2)/(1-np.cos(self.acceptance_angle*np.pi/180/2/M))

        # Sigma is chosen so that Gaussian wears out at a distance
        # equal to interommatidial distance
        self.sigma = np.tan(self.acceptance_angle*np.pi/180/2)

        self.dxy = np.diff(self.grid[0][0, :2]) * \
            np.diff(self.grid[1][:2, 0])[0]*self.radius
        self.parameter_set = True

        super(Cylinder_Gaussian_RF, self).load_parameters()

    @property
    def refa(self):
        try:
            return self.refz
        except:
            print('RF parameters were not loaded correctly')
            raise

    @property
    def refb(self):
        try:
            return self.reftheta
        except:
            print('RF parameters were not loaded correctly')
            raise

    def _call_filter_func(self, N_filters, startbias):
        itemsize = self.dtype.itemsize
        self.filter_func.prepared_call(
            self._nblocks,
            self._nthreads,
            self.filters.gpudata,
            self.filters.ld,
            self.d_grid[0].gpudata,
            self.d_grid[1].gpudata,
            int(self.d_refz.gpudata) + startbias * itemsize,
            int(self.d_reftheta.gpudata) + startbias * itemsize,
            self.size, N_filters,
            self.dxy, self.radius, self.kappa, self.sigma)


context_dependent_memoize
def _get_von_mises_fisher_rf(dtype):
    template = """

#define ONE_OVER_TWO_PI 0.159154943091895

__global__ void
von_mises_fisher(%(type)s* rfs, int rfs_ld,
                 %(type)s* elevs, %(type)s* azims,
                 %(type)s* refelevs, %(type)s* refazims,
                 int grid_size, int num_photor,
                 %(type)s delev_dazim, %(type)s kappa)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    %(type)s elevation = 0;
    %(type)s azimuth = 0;
    %(type)s s1 = 0;
    %(type)s c1 = 0;

    if (tid < grid_size)
    {
        elevation = elevs[tid];
        azimuth = azims[tid];
        sincos%(fletter)s(elevation, &s1, &c1);

        %(type)s s2, c2, refazim, inner_minus_1;

        for(int i = 0; i < num_photor; ++i)
        {
            sincos%(fletter)s(refelevs[i], &s2, &c2);
            refazim = refazims[i];

            inner_minus_1 = c1*c2*cos%(fletter)s(refazim-azimuth)+s1*s2-1;
            // take kappa to 0.75 power and multiply by 3.747 is to
            // normalize integral of the squared rf to 1
            rfs[i*rfs_ld + tid] = kappa * ONE_OVER_TWO_PI
                                  / (1-exp%(fletter)s(-2*kappa))
                                  * exp%(fletter)s(kappa*inner_minus_1)
                                  * delev_dazim * c1 ;

        }
    }

}
"""
    # double
    # ptxas info    : 0 bytes gmem, 272 bytes cmem[2]
    # ptxas info    : Compiling entry function 'von_mises_fisher' for 'sm_20'
    # ptxas info    : Function properties for von_mises_fisher
    #    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    # ptxas info    : Used 42 registers, 104 bytes cmem[0], 52 bytes cmem[16]
    # ptxas info    : Function properties for __internal_trig_reduction_slowpathd
    # 40 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    vtype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                       "fletter": "f" if vtype == np.float32 else ""},
                       options=["--ptxas-options=-v"])
    func = mod.get_function("von_mises_fisher")
    func.prepare([np.intp, np.int32, np.intp, np.intp, np.intp,
                  np.intp, np.int32, np.int32, vtype, vtype])
    func.set_cache_config(cuda.func_cache.PREFER_L1)
    return func


def _get_gaussian_cylinder(dtype):
    template = """

#include <stdio.h>

#define PI 3.14159265358979
#define ONE_OVER_TWO_PI 0.159154943091895
#define SQRT_ONE_OVER_TWO_PI 0.3989422804

__global__ void
gaussian_cylinder1(%(type)s* rfs, int rfs_ld,
                   %(type)s* zs, %(type)s* thetas,
                   %(type)s* ref_zs, %(type)s* ref_thetas,
                   int grid_size, int num_photor,
                   %(type)s dz_dtheta, %(type)s radius,
                   %(type)s kappa, %(type)s sigma)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    %(type)s z, theta;
    %(type)s f_z, f_theta, ref_z, ref_theta;
    %(type)s atan_z, atan_refz, tan_zMzref;


    if(tid < grid_size)
    {

        z = zs[tid];
        theta = thetas[tid];

        atan_z = atan2%(fletter)s(z,radius);

        for(int i = 0; i < num_photor; ++i)
        {
            /* exp(-((tan(arctan(z)-arctan(z0))**2)/2*sigma**2))
               /(sigma*sqrt(2pi))/(z**2+1)*dz */
            /* exp(kappa*cos(theta-theta0))*/
            ref_z = ref_zs[i];
            ref_theta = ref_thetas[i];

            atan_refz = atan2%(fletter)s(ref_z, radius);

            if (fabs%(fletter)s(atan_z - atan_refz) < PI/2) {
                tan_zMzref = tan%(fletter)s(atan_z - atan_refz)/sigma;
                f_z = SQRT_ONE_OVER_TWO_PI * exp%(fletter)s(-tan_zMzref*tan_zMzref)
                        /(z*z+1)/sigma;
                f_theta = exp%(fletter)s(kappa*(cos%(fletter)s(theta-ref_theta)-1));
                rfs[i*rfs_ld + tid] = f_z*f_theta*dz_dtheta;
            } else {
                rfs[i*rfs_ld + tid] = 0;
            }
            /* TODO if ever used output should be normalized */
        }
    }
}

/* sigma is not used but is not needed either */
__global__ void
gaussian_cylinder2(%(type)s* rfs, int rfs_ld,
                   %(type)s* zs, %(type)s* thetas,
                   %(type)s* ref_zs, %(type)s* ref_thetas,
                   int grid_size, int num_photor,
                   %(type)s dz_dtheta, %(type)s radius,
                   %(type)s kappa, %(type)s sigma)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    %(type)s x, y, z;
    %(type)s ref_x, ref_y, ref_z;
    %(type)s norm_x, norm_y, norm_z;
    %(type)s inv_len, inv_ref_len, inp;


    if(tid < grid_size)
    {

        /* variables transformation */
        sincos%(fletter)s(thetas[tid], &y, &x);
        x *= radius;
        y *= radius;
        z = zs[tid];

        inv_len = 1/sqrt(radius*radius + z*z);

        norm_x = x*inv_len;
        norm_y = y*inv_len;
        norm_z = z*inv_len;

        for(int i = 0; i < num_photor; ++i)
        {
            /* reference variables transformation */
            sincos%(fletter)s(ref_thetas[i], &ref_y, &ref_x);
            ref_x *= radius;
            ref_y *= radius;
            ref_z = ref_zs[i];

            inv_ref_len = 1/sqrt(ref_x*ref_x + ref_y*ref_y + ref_z*ref_z);

            ref_x *= inv_ref_len;
            ref_y *= inv_ref_len;
            ref_z *= inv_ref_len;

            inp = norm_x*ref_x + norm_y*ref_y + norm_z*ref_z;

            rfs[i*rfs_ld + tid] = kappa*ONE_OVER_TWO_PI
                                  /(1-exp%(fletter)s(-2*kappa))
                                  *exp%(fletter)s(kappa*(inp-1))*radius
                                  *(inv_len*inv_len*inv_len) * dz_dtheta;
        }
    }
}
"""
    vtype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                       "fletter": "f" if vtype == np.float32 else ""},
                       options=["--ptxas-options=-v"])
    func = mod.get_function("gaussian_cylinder2")
    func.prepare([np.intp, np.int32, np.intp, np.intp, np.intp,
                  np.intp, np.int32, np.int32, vtype, vtype, vtype, vtype])
    func.set_cache_config(cuda.func_cache.PREFER_L1)
    return func

if __name__ == "__main__":
    import pycuda.autoinit
    import numpy.random as random

    random.seed(481988)

    dtype = np.double
    filter_func = _get_gaussian_cylinder(dtype)

    # Constants
    S1 = 128
    S2 = 128
    PHOTORECEPTORS = 8
    M_size = S1*S2  # same as grid[0].size
    N_filters = PHOTORECEPTORS

    RAD = 1
    KAPPA = 20
    SIGMA = 1  # or angle
    NTHREADS = (128, 1, 1)
    NBLOCKS = ((M_size-1) // NTHREADS[0] + 1, 1)

    d_filters = parray.empty((N_filters, M_size), dtype)
    grid = np.meshgrid(np.linspace(-1, 1, num=S1),
                       np.linspace(-np.pi, np.pi, num=S2))
    d_grid = [parray.to_gpu(grid[i].flatten()) for i in range(len(grid))]

    dxy = np.diff(grid[0][0, :2])*np.diff(grid[1][:2, 0])[0]

    ref_z = 2*random.rand(PHOTORECEPTORS)-1  # -1 to 1
    d_refz = parray.to_gpu(ref_z)

    ref_theta = np.pi*random.rand(PHOTORECEPTORS)  # half cylinder
    d_reftheta = parray.to_gpu(ref_theta)
    filter_func.prepared_call(
        NBLOCKS,
        NTHREADS,
        d_filters.gpudata,
        d_filters.ld,
        d_grid[0].gpudata,
        d_grid[1].gpudata,
        d_refz.gpudata,
        d_reftheta.gpudata,
        M_size, N_filters,
        dxy, RAD, KAPPA, SIGMA)

    h_filters = d_filters.get()
    np.set_printoptions(precision=4)
    print('Sums of filter functions:\n{}\n at points:\n*z:\n{}\n*theta\n{}\n'
          'and output shape:{}'
          .format(np.sum(h_filters, axis=1),
                  ref_z, ref_theta, h_filters.shape))
