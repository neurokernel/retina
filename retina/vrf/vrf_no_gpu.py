from abc import ABCMeta, abstractmethod, abstractproperty

import math

import numpy as np


class RF(object):
    __metaclass__ = ABCMeta

    def __init__(self, grid):
        """
            grid: a meshgrid list with arrays for 2 coordinates
                  represents the screen coordinates
        """

        if grid[0].ndim == 1:
            grid = np.meshgrid(grid[0], grid[1])
        else:
            grid = grid

        self.grid = grid
        self.grid0 = self.grid[0].reshape(-1)
        self.grid1 = self.grid[1].reshape(-1)

        self.size = grid[0].size
        self.dtype = np.dtype(np.double)

        # flags
        self.parameters_loaded = False

    @abstractmethod
    def load_parameters(self, **kwargs):
        '''
            should set num_neurons and
            parameters_loaded flag
        '''
        pass

    @abstractproperty
    def refa(self):
        pass

    @abstractproperty
    def refb(self):
        pass

    def compute_filters(self):
        # there is an exception when this object is too large
        # so lower precision was used
        filters = np.empty((self.size, self.num_neurons), dtype=np.float32)
        for i in range(self.num_neurons):
            filters[:, i] = self._generate_filter(i)
        self.filters = filters

    def filter(self, video_input):
        """
        Performs RF filtering on input video
        for all the rfs
        """
        # video dimensions should match screen dimensions
        # numpy resize operation doesn,t make any checks
        if len(video_input.shape) == 2:
            # if input has 2 dimensions
            assert video_input.shape[1] == self.size
        else:
            # if input has 3 dimensions
            assert (video_input.shape[1]*video_input.shape[2] ==
                    self.size)

        # rasterizing inputs
        video_input.resize((video_input.shape[0], self.size))

        output = np.empty((video_input.shape[0], self.num_neurons))

        for i in range(self.num_neurons):
            filters = self._generate_filter(i)
            output[:, i] = np.dot(video_input, filters)
        return output

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
        image_input.resize((1,self.size))

        return np.dot(image_input, self.filters)


class Sphere_Gaussian_RF(RF):
    ONE_OVER_TWO_PI = 0.159154943091895

    def __init__(self, grid):
        super(Sphere_Gaussian_RF, self).__init__(grid)

    @property
    def refa(self):
        return self.refelev

    @property
    def refb(self):
        return self.refazim

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
        self.compute_filters()

        self.parameter_set = True

    def _generate_filter(self, i):
        refelev = self.refelev[i]
        refazim = self.refazim[i]

        elevs = self.grid0
        azims = self.grid1
        npelevs = np.cos(elevs)
        innerM1 = npelevs*np.cos(refelev)*np.cos(refazim-azims) \
            + np.sin(elevs)*np.sin(refelev)-1

        return self.kappa * self.ONE_OVER_TWO_PI / (1-np.exp(-2*self.kappa)) * \
            np.exp(self.kappa*innerM1) * self.dxy * np.cos(azims)


class Cylinder_Gaussian_RF(RF):
    ONE_OVER_TWO_PI = 0.159154943091895

    def __init__(self, grid):
        super(Cylinder_Gaussian_RF, self).__init__(grid)

    @property
    def refa(self):
        return self.refz

    @property
    def refb(self):
        return self.reftheta

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
        self.compute_filters()

        self.parameter_set = True

    def _generate_filter(self, i):
        radius = self.radius

        #
        reftheta = self.reftheta[i]
        refz = self.refz[i]

        refx = math.sin(reftheta) * radius
        refy = math.cos(reftheta) * radius

        inv_ref_len = 1/math.sqrt(radius*radius + refz*refz)
        refx *= inv_ref_len
        refy *= inv_ref_len
        refz *= inv_ref_len

        #
        thetas = self.grid[1].reshape(-1)
        zs = self.grid[0].reshape(-1)

        xs = np.sin(thetas) * radius
        ys = np.cos(thetas) * radius

        inv_len = 1/math.sqrt(radius*radius + zs*zs)
        xs *= inv_len
        ys *= inv_len
        zs *= inv_len

        #
        inp = xs*refx + ys*refy + zs*refz

        return self.kappa * self.ONE_OVER_TWO_PI / (1-np.exp(-2*self.kappa)) * \
            np.exp(self.kappa*(inp-1))*radius * (inv_len*inv_len*inv_len) * \
            self.dxy

