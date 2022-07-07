from __future__ import division

import numpy as np

from scipy.interpolate import RectBivariateSpline

from .signaltransform import SignalTransform


class ImageTransform(SignalTransform):
    def __init__(self, original_grid, new_grid):
        """
        :param original_grid: list of 2 1D arrays of original coordinates
                              in strictly ascending order
        :param new_grid: list of 2 arrays of new coordinates
        """
        self.ogrid = original_grid
        self.ngrid = new_grid

    def interpolate(self, images):
        new_images = np.empty(
            (images.shape[0],)+ self.ngrid[0].shape, images.dtype)
        for i, image in enumerate(images):
            new_images[i] = self.interpolate_individual(image)
        return new_images

    def interpolate_individual(self, image):
        # unpacking
        ogridx, ogridy = self.ogrid
        ngridx, ngridy = self.ngrid
    
        
        # put y first since the 1st axis of array is the y not x
        f = RectBivariateSpline(ogridy, ogridx, image, kx=1, ky=1)
        f_new = f.ev(ngridy.flatten(), ngridx.flatten()).reshape(ngridx.shape)
        return f_new

