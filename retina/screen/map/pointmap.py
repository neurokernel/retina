from abc import ABCMeta, abstractmethod


class PointMap(object):
    """ Interface of mapping a point from one surface to another
        (hence the 2 parameters)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, p1, p2):
        """ map of point (p1, p2) from one surface to another """
        return

    @abstractmethod
    def invmap(self, p1, p2):
        """ inverse map of point (p1, p2) from one surface to another """
        return


class PointMapWithDirection(object):
    """ Interface of mapping a point from one surface to another
        (hence the 2 parameters)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, p1, p2, dp1, dp2):
        """ map of point (p1, p2) from one surface to another
            in direction (dp1, dp2) """
        return

    @abstractmethod
    def invmap(self, p1, p2, dp1, dp2):
        """ inverse map of point (p1, p2) from one surface to another
            in direction (dp1, dp2) """
        return


class EyeToImagePointMap(PointMap):
    """ Encapsulates 2 PointMap transformations to map a point
        from the fly's eye to a screen and then to an image
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map_screen_to_image(self, p1, p2):
        mapscreen = self._map_screen

        r1, r2 = mapscreen.map(p1, p2)
        return (r1, r2)

    @abstractmethod
    def invmap_screen_to_image(self, p1, p2):
        mapscreen = self._map_screen

        r1, r2 = mapscreen.invmap(p1, p2)
        return (r1, r2)

    @abstractmethod
    def map_eye_to_screen(self, p1, p2):
        mapeye = self._map_eye

        r1, r2 = mapeye.map(p1, p2)
        return (r1, r2)

    @abstractmethod
    def invmap_eye_to_screen(self, p1, p2):
        mapeye = self._map_eye

        r1, r2 = mapeye.invmap(p1, p2)
        return (r1, r2)

    # implementation of superclass abstract classes
    def map(self, p1, p2):
        mapeye = self._map_eye
        mapscreen = self._map_screen

        t1, t2 = mapeye.map(p1, p2)
        r1, r2 = mapscreen.map(t1, t2)
        return (r1, r2)

    def invmap(self, p1, p2):
        mapeye = self._map_eye
        mapscreen = self._map_screen

        t1, t2 = mapscreen.invmap(p1, p2)
        r1, r2 = mapeye.invmap(t1, t2)
        return (r1, r2)
