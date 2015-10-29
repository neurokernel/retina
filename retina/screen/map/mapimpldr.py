from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
PI = np.pi

from pointmap import PointMapWithDirection


def solve_quadratic(a, b, c):
    """
        b is b/2 and computed D is D/4
        so solutions are (-b +/- sqrt(D))/a
        the solutions with the lowest absolute
        values are returned

        a,b,c can be arrays

    """

    D = b*b - a*c
    # select the minimum of the 2 roots
    # so the closest intersection point will be chosen
    l12 = np.array([(-b + np.sqrt(D))/a,
                    (-b - np.sqrt(D))/a]).reshape(2, -1)

    indxl = np.argmin(abs(l12), axis=0)
    indyl = np.indices(indxl.shape)[0]

    # return the solution(s) with the lowest absolute value
    return l12[indxl, indyl]


class SphereToSphereMap(PointMapWithDirection):
    def __init__(self, r1, r2):
        self._r1 = r1
        self._r2 = r2

    @classmethod
    def from_retina_screen(cls, retina, screen):
        return cls(retina.radius, screen.radius)

    @staticmethod
    def map_aux(elev, azim, delev, dazim, r1, r2):
        """ In case of 2 points of intersection picks the closest
        """
        # Transform from spherical to Cartesian coordinates
        x1 = -r1*np.cos(elev)*np.cos(azim)
        y1 = -r1*np.cos(elev)*np.sin(azim)
        z1 = r1*np.sin(elev)

        dx1 = -np.cos(delev)*np.cos(dazim)
        dy1 = -np.cos(delev)*np.sin(dazim)
        dz1 = np.sin(delev)

        result_shape = x1.shape

        # Solving
        # ||x + l*dx||_2 = r2
        a = 1
        b = x1*dx1 + y1*dy1 + z1*dz1
        c = r1*r1 - r2*r2

        l = solve_quadratic(a, b, c).reshape(result_shape)

        x2 = x1 + l*dx1
        y2 = y1 + l*dy1
        z2 = z1 + l*dz1

        map_elev = np.arcsin(z2/r2)
        map_azim = np.arctan2(-y2, -x2)

        try:
            return (map_elev.reshape(result_shape),
                    map_azim.reshape(result_shape))
        except AttributeError:  # if elev and azim scalars
            return (np.asscalar(map_elev), np.asscalar(map_azim))

    def map(self, elev, azim, delev, dazim):
        return self.map_aux(elev, azim, delev, dazim,
                            self._r1, self._r2)

    def invmap(self, elev, azim, delev, dazim):
        return self.map_aux(elev, azim, delev, dazim,
                            self._r2, self._r1)


class SphereToCylinderMap(PointMapWithDirection):
    MIN_LEN = np.tan(PI/180)

    """ Map points of a sphere to a cylinder.
        The axon of the cylinder goes through the center of the sphere.
    """
    def __init__(self, r1, r2):
        """ r1: sphere radius
            r2: cylinder radius
        """
        self._r1 = r1
        self._r2 = r2

    @classmethod
    def from_retina_screen(cls, retina, screen):
        return cls(retina.radius, screen.radius)

    def map(self, elev, azim, delev, dazim):
        """
            * Not accurate taken from respective method in mapimpl.py
            z = 0 corresponds to azimuth 0 or pi
            z = max at azimuth -pi/2

            axes were chosen that way so that eye faces the screen

            elev: the elevation of sphere, can be scalar or numpy array
            azim: the azimuth of sphere, can be scalar or numpy array

            returns: tuple of arrays of the same size as the initial ones
                     (that means even if input is scalar)
                     representing respective theta and z coordinates on cylinder
        """
        r1 = self._r1
        r2 = self._r2

        # Transform from spherical to Cartesian coordinates
        x1 = -r1*np.cos(elev)*np.cos(azim)
        y1 = -r1*np.cos(elev)*np.sin(azim)
        z1 = r1*np.sin(elev)

        dx1 = -np.cos(delev)*np.cos(dazim)
        dy1 = -np.cos(delev)*np.sin(dazim)
        dz1 = np.sin(delev)

        result_shape = x1.shape

        # solving
        # ||(x,y) + l*(dx,dy)||_2 = r2
        a = dx1*dx1 + dy1*dy1
        a[a < self.MIN_LEN] = self.MIN_LEN
        b = x1*dx1 + y1*dy1
        c = x1*x1 + z1*z1 - r2*r2

        l = solve_quadratic(a, b, c).reshape(result_shape)

        x2 = x1 + l*dx1
        y2 = y1 + l*dy1
        z2 = z1 + l*dz1

        map_z = z2
        map_theta = np.arctan2(y2, x2)

        try:
            return (map_z.reshape(result_shape),
                    map_theta.reshape(result_shape))
        except AttributeError:  # if z and theta scalars
            return (np.asscalar(map_z), np.asscalar(map_theta))

    def invmap(self, z, theta):
        """
            Similar to map, but happens from cylinder to sphere
        """
        delev, dazim = self._direction
        r1 = self._r1
        r2 = self._r2

        # Transform from cylindrical to Cartesian coordinates
        x1 = r2*np.cos(theta)
        y1 = r2*np.sin(theta)
        z1 = z

        # Transform from spherical to Cartesian coordinates
        dx1 = -np.cos(delev)*np.cos(dazim)
        dy1 = -np.cos(delev)*np.sin(dazim)
        dz1 = np.sin(delev)


        result_shape = x1.shape

        # solving
        # ||(x,y,z) + l*(dx,dy,dz)||_2 = r1
        a = 1  # dx1*dx1+dy1*dy1+dz1*dz1
        b = x1*dx1 + y1*dy1 + z1*dz1
        c = x1*x1 + y1*y1 + z1*z1 - r1*r1

        l = solve_quadratic(a, b, c).reshape(result_shape)

        x2 = x1 + l*dx1
        y2 = y1 + l*dy1
        z2 = z1 + l*dz1

        map_elev = np.arcsin(z2/r1)
        map_azim = np.arctan2(-y2, -x2)

        try:
            return (map_elev.reshape(result_shape),
                    map_azim.reshape(result_shape))
        except AttributeError:  # if elev and azim scalars
            return (np.asscalar(map_elev), np.asscalar(map_azim))

