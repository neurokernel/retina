from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np

from .pointmap import PointMap, EyeToImagePointMap

PI = np.pi
MYFLOAT = np.float64


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


class AlbersProjectionMap(PointMap):
    """ https://en.wikipedia.org/wiki/Albers_projection
        Compared to the entry of 8/21/2015
        (numbers were chosen to have simpler formulas)
        phi1 = 90, phi2 = 0, phi0 = 90, lambda0 = 90
        lambda <- 2n (lambda - lambda0) (doubles the area)
        =>
        n = 0.5
        rho0 = 0, C = 1, rho = 2sqrt(1-sin(phi))
    """

    def __init__(self, r, eulerangles=None):
        """ r: radious of the sphere from which the projection is made
        """
        self._r = r
        self.rotationmap = EulerAnglesMap(eulerangles)
        self.x_rotationmap = EulerAnglesMap(eulerangles=[0, np.pi/2, 0])

    def map(self, elev, azim):
        """ Returns (nan, nan) if point cannot be mapped else (x, y)
            arguments can be numpy arrays or scalars
        """
        elev, azim = self.rotationmap.invmap(elev, azim)
        elev, azim = self.x_rotationmap.invmap(elev, azim)

        rxy = self._r*np.sqrt(1-np.sin(elev))
        x = -rxy*np.cos(azim)
        y = -rxy*np.sin(azim)
        return (x, y)

    def invmap(self, x, y):
        """ Returns (nan, nan) if point cannot be mapped else
            (elevation, azimuth)
            arguments can be numpy arrays or scalars
        """
        r = self._r
        rxy = np.sqrt(x*x + y*y)

        elev = np.arcsin(1-(rxy/r)**2)
        azim = np.arctan2(-y, -x)

        # this angle is nan if rxy = 0 but can be anything
        try:
            azim[np.isnan(azim)] = 0
        except TypeError:  # Thrown if azim is scalar
            if np.isnan(azim):
                azim = 0

        try:
            azim[np.isnan(elev)] = np.nan
        except TypeError:  # Thrown if azim is scalar
            if np.isnan(elev):
                azim = np.nan

        elev, azim = self.x_rotationmap.map(elev, azim)
        return self.rotationmap.map(elev, azim)


class EquidistantProjectionMap(PointMap):
    """ https://en.wikipedia.org/wiki/Equidistant_conic_projection
    """
    def __init__(self, r=1):
        """ r: radious of the sphere from which the projection is made
        """
        self._r = r

    def map(self, elev, azim):
        """ Returns (nan, nan) if point cannot be mapped else (x, y)
            arguments can be numpy arrays or scalars
        """
        rxy = self._r*(1 - elev/(PI/2))
        x = -rxy*np.cos(azim)
        y = -rxy*np.sin(azim)
        return (x, y)

    def invmap(self, x, y):
        """ Returns (nan, nan) if point cannot be mapped else
            (elevation, azimuth)
            arguments can be numpy arrays or scalars
        """
        r = self._r
        rxy = np.sqrt(x*x + y*y)

        elev = (1 - rxy/r)*(PI/2)
        azim = np.arctan2(-y, -x)

        return (elev, azim)


class EulerAnglesMap(PointMap):
    def __init__(self, eulerangles=None):
        self._eulerangles = eulerangles

    @staticmethod
    def rotx(p, angle):
        cosa = np.cos(angle)
        sina = np.sin(angle)
        return np.dot(np.array([[1, 0, 0],
                                [0, cosa, -sina],
                                [0, sina, cosa]]), p)

    @staticmethod
    def rotz(p, angle):
        cosa = np.cos(angle)
        sina = np.sin(angle)
        return np.dot(np.array([[cosa, -sina, 0],
                                [sina, cosa, 0],
                                [0, 0, 1]]), p)

    @classmethod
    def map_aux(cls, elev0, azim0, a, b, c):
        try:
            elev0_flat = elev0.flatten()
            azim0_flat = azim0.flatten()
        except AttributeError:  # if variables are scalars
            elev0_flat = elev0
            azim0_flat = azim0

        p0 = np.array([np.cos(elev0_flat)*np.cos(azim0_flat),
                       -np.cos(elev0_flat)*np.sin(azim0_flat),
                       np.sin(elev0_flat)])

        p1 = cls.rotz(p0, a)
        p2 = cls.rotx(p1, b)
        p3 = cls.rotz(p2, c)

        elev3 = np.arcsin(p3[2])
        azim3 = np.arctan2(-p3[1], p3[0])
        try:
            azim3[azim3 < 0] = azim3[azim3 < 0] + 2*np.pi
        except:
            if azim3 < 0:
                azim3 += 2*np.pi

        try:
            return elev3.reshape(elev0.shape), azim3.reshape(azim0.shape)
        except:
            return elev3, azim3

    def map(self, elev0, azim0):
        if self._eulerangles is None:
            return elev0, azim0
        else:
            phi, theta, psi = self._eulerangles

        return self.map_aux(elev0, azim0, phi, theta, psi)


    def invmap(self, elev0, azim0):
        if self._eulerangles is None:
            return elev0, azim0
        else:
            phi, theta, psi = self._eulerangles

        return self.map_aux(elev0, azim0, -psi, -theta, -phi)


class SphereToSphereMap(PointMap):
    """ Map points from one sphere to another.
        spheres are cocentered
    """
    def __init__(self, r1, r2, direction):
        """ r1: initial sphere radius
            r2: projected sphere radius
            direction: can be a tuple or a list of 2 elements
                       elevation, azimuth
        """
        self._r1 = r1
        self._r2 = r2
        self._direction = direction

    @staticmethod
    def map_aux(elev, azim, direction, r1, r2):
        """ In case of 2 points of intersection picks the closest
        """

        delev, dazim = direction

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
        except AttributeError:
            # if elev and azim scalars
            return (np.asscalar(map_elev), np.asscalar(map_azim))

    def map(self, elev, azim):
        """ Returns closest point of intersection or nan if there is none
            elev: the elevation of sphere 1, can be scalar or numpy array
            azim: the azimuth of sphere 1, can be scalar or numpy array

            returns: tuple of arrays of the same size as the initial ones
                     (that means even if input is scalar)
                     representing respective coordinates on sphere 2
        """
        r1 = self._r1
        r2 = self._r2
        direction = self._direction

        return self.map_aux(elev, azim, direction, r1, r2)

    def invmap(self, elev, azim):
        """ Similar to map, but happens from sphere 2 to 1
        """
        r1 = self._r1
        r2 = self._r2
        direction = self._direction

        return self.map_aux(elev, azim, direction, r2, r1)


class SphereToCylinderMap(PointMap):
    MIN_LEN = np.tan(PI/180)

    """ Map points of a sphere to a cylinder.
        The axon of the cylinder goes through the center of the sphere.
    """
    def __init__(self, r1, r2, direction):
        """ r1: sphere radius
            r2: cylinder radius
            direction: can be a tuple or a list of 2 elements
                       elevation, azimuth
        """
        self._r1 = r1
        self._r2 = r2
        self._direction = direction

    def map(self, elev, azim):
        """
            z = 0 corresponds to azimuth 0 or pi
            z = max at azimuth -pi/2

            axes were chosen that way so that eye faces the screen

            elev: the elevation of sphere, can be scalar or numpy array
            azim: the azimuth of sphere, can be scalar or numpy array

            returns: tuple of arrays of the same size as the initial ones
                     (that means even if input is scalar)
                     representing respective theta and z coordinates on cylinder
        """

        delev, dazim = self._direction
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
        try:
            a[a < self.MIN_LEN] = self.MIN_LEN
        except TypeError:  # if scalar
            a = min(a, self.MIN_LEN)
        b = x1*dx1 + y1*dy1
        c = x1*x1 + y1*y1 - r2*r2

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


class CylinderToImageMap(PointMap):
    """
        Performs a mapping from a cylinder to a plane and back
    """

    def __init__(self, L):
        """
            L: image of length L(in x direction) will be mapped
                to half the cylinder the whole cylinder can be used if
                L=image_L/2

            TODO maybe parameters for height will be needed later
        """
        self._L = L

    def map(self, z, theta):
        """
            z: -M to M, M depends on application
            theta: 0 to pi

            y direction of image is aligned with z direction of cylinder
        """
        return (self._L/2 - self._L*theta/PI, z)

    def invmap(self, x, y):
        """
            x: -M to M, M depends on application
            y: -L/2 to L/2

            y direction of image is aligned with z direction of cylinder
        """
        return (y, PI*(0.5-x/self._L))


class EyeToSphereToImageMap(EyeToImagePointMap):
    """
        Performs 2 mappings. From eye hemisphere to a hemisphere screen
        and then on a plane.
    """
    # static variables (don't change) except _map_eye
    MERIDIANS = 800
    PARALLELS = 50

    _reye = 1
    _rscreen = 10
    # maps eye to screen and inversely
    _map_eye = None
    # maps screen to plane and inversely
    _map_screen = AlbersProjectionMap(_rscreen)

    def __init__(self, direction):
        self._map_eye = SphereToSphereMap(self._reye, self._rscreen, direction)

    def map(self, eyeelev, eyeazim):
        return super(EyeToSphereToImageMap, self).map(eyeelev, eyeazim)

    def invmap(self, x, y):
        return super(EyeToSphereToImageMap, self).invmap(x, y)

    def map_eye_to_screen(self, eyeelev, eyeazim):
        return super(EyeToSphereToImageMap, self) \
            .map_eye_to_screen(eyeelev, eyeazim)

    def invmap_eye_to_screen(self, screenelev, screenazim):
        return super(EyeToSphereToImageMap, self) \
            .invmap_eye_to_screen(screenelev, screenazim)

    def map_screen_to_image(self, screenelev, screenazim):
        return super(EyeToSphereToImageMap, self) \
            .map_screen_to_image(screenelev, screenazim)

    def invmap_screen_to_image(self, x, y):
        return super(EyeToSphereToImageMap, self).invmap_screen_to_image(x, y)

    # depends only on how images are mapped on screen which is an attribute of
    # the class
    @classmethod
    def get_intensities(cls, image, config, photorelev, photorazim,
                        nrings, nommatidia):
        return super(EyeToSphereToImageMap, cls).get_intensities(
            image, config, photorelev, photorazim, nrings, nommatidia)


class EyeToCylinderToImageMap(EyeToImagePointMap):
    """
        Performs 2 mappings. From eye hemisphere to a cylinder screen
        and then on a plane.
    """

    # static variables (don't change) except _map_eye
    COLUMNS = 800
    PARALLELS = 50

    _reye = 1
    _rscreen = 10
    # maps eye to screen and inversely
    _map_eye = None
    # maps screen to plane and inversely
    _map_screen = CylinderToImageMap(2*_rscreen)

    def __init__(self, direction):
        self._map_eye = SphereToCylinderMap(self._reye, self._rscreen,
                                            direction)

    def map(self, eyeelev, eyeazim):
        return super(EyeToCylinderToImageMap, self).map(eyeelev, eyeazim)

    def invmap(self, x, y):
        return super(EyeToCylinderToImageMap, self).invmap(x, y)

    def map_eye_to_screen(self, eyeelev, eyeazim):
        return super(EyeToCylinderToImageMap, self) \
            .map_eye_to_screen(eyeelev, eyeazim)

    def invmap_eye_to_screen(self, screenz, screentheta):
        return super(EyeToCylinderToImageMap, self) \
            .invmap_eye_to_screen(screenz, screentheta)

    def map_screen_to_image(self, screenz, screentheta):
        return super(EyeToCylinderToImageMap, self) \
            .map_screen_to_image(screenz, screentheta)

    def invmap_screen_to_image(self, x, y):
        return super(EyeToCylinderToImageMap, self) \
            .invmap_screen_to_image(x, y)

    # depends only on how images are mapped on screen which is an attribute of
    # the class
    @classmethod
    def get_intensities(cls, image, config, photorelev, photorazim,
                        nrings, nommatidia):
        return super(EyeToCylinderToImageMap, cls).get_intensities(
            image, config, photorelev, photorazim, nrings, nommatidia)


def pointmapfactory(map_type):
    # I think this implementation will find only the classes defined in this
    # file
    all_pointmap_cls = PointMap.__subclasses__()
    all_pointmap_names = [cls.__name__ for cls in all_pointmap_cls]
    try:
        pointmap_cls = all_pointmap_cls[all_pointmap_names.index(map_type)]
    except ValueError:
        print('Invalid PointMap subclass name:{}'.format(map_type))
        return None
    return pointmap_cls
