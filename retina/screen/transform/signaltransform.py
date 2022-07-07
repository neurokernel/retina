from abc import ABCMeta, abstractmethod


class SignalTransform(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def interpolate(self, image):
        """ gets the value of the signal at the specific point,
            as the signal is discrete it will be interpolated
        """
        return
