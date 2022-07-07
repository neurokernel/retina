from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np


class PhotorInput(object):
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.photoreceptors = config['Photoreceptor']['photoreceptors']

        ph_config = config['PhotorInputType']
        np.random.seed(ph_config['seed'])
        self.add_noise = ph_config['add_noise']
        self.noise_var = ph_config['noise_var']
        self.same_inputs = ph_config['same_inputs']

    @abstractmethod
    def get_input(self, num_steps):
        pass

    def get_input_config(self, config):
        intype = config['Photoreceptor']['single_intype']
        return config['PhotorInputType'][intype]

    def add_gwn(self, inputs):
        if self.add_noise:
            return inputs * \
                (1 + self.noise_var*np.random.randn(*inputs.shape)).clip(0)
        else:
            return inputs

    def make_same(self, inputs):
        if self.same_inputs:
            return np.tile(inputs[0, :], (self.photoreceptors, 1))
        else:
            return inputs



class Image(PhotorInput):
    def __init__(self, config):
        super(Image, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.image_file = config['image_file']
        self.scale = config['scale']
        self.image = self.readimage()
        self.flat_im = self.image[:100, :100].flatten()

        self.flat_im *= self.scale/self.flat_im.max()

    def readimage(self):
        from scipy.io import loadmat

        try:
            mat = loadmat(self.image_file)
        except AttributeError:
            print('Tried to read image before setting the file variable')
            raise
        except IOError:
            if self.image_file is None:
                print('Image file not specified')
            raise

        try:
            return np.array(mat['im'])
        except KeyError:
            print('No variable "im" in given mat file')
            print('Available variables (and meta-data): {}'
                  .format(mat.keys()))
            raise

    def get_input(self, num_steps):
        num_photors = self.photoreceptors

        im_size = self.flat_im.size
        reps = 1 + num_steps // im_size

        tmp = np.tile(self.flat_im, (num_photors, reps))[:, :num_steps]
        return self.make_same(self.add_gwn(tmp)).T

    def get_flat_image(self):
        num_photors = self.photoreceptors
        tmp = np.tile(self.flat_im, (num_photors, 1))
        return self.make_same(self.add_gwn(tmp)).T


class Series(PhotorInput):
    def __init__(self, config):
        super(Series, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.input_file = config['input_file']
        self.scale = config['scale']
        self.series = self.readseries()

    def readseries(self):
        arr = []
        with open(self.input_file, 'r') as f:
            for line in f:
                arr.append(float(line))

        return self.scale*np.array(arr)

    def get_input(self, num_steps):
        num_photors = self.photoreceptors
        im_size = self.series.size
        reps = 1 + num_steps // im_size

        tmp = np.tile(self.series, (num_photors, reps))[:, :num_steps]
        return self.make_same(self.add_gwn(tmp)).T


class StepSeq(PhotorInput):
    def __init__(self, config):
        super(StepSeq, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.levels = tuple(config['levels'])
        self.step_size = config['step_size']
        self.step_freq = config['step_freq']

    def get_input(self, num_steps):
        num_photors = self.photoreceptors

        # reference low high
        r, l, h = self.levels
        step_size = self.step_size
        step_freq = self.step_freq

        arr = r * np.ones(num_steps)

        ind = np.random.poisson(1/step_freq)
        while(ind < num_steps):
            arr[ind:ind+step_size] = l if np.random.uniform()<0.5 else h
            ind += max(1+step_size, np.random.poisson(1/step_freq))

        tmp = np.tile(arr, (num_photors, 1))
        return self.make_same(self.add_gwn(tmp)).T


class Steps(PhotorInput):
    def __init__(self, config):
        super(Steps, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.levels = config['levels']
        self.step_size = config['step_size']
        self.scale = config['scale']
        self.multipliers = config['multipliers']

    def get_input(self, num_steps):
        num_photors = self.photoreceptors
        levels = self.levels
        start, stop, num_mult = self.multipliers
        scale = self.scale
        step_size = self.step_size

        time_reps = 1 + num_steps // step_size*len(levels)
        photor_reps = 1 + num_photors // num_mult

        if scale == 'log':
            scaling_range = np.logspace(start, stop, num_mult)
        else:
            scaling_range = np.linspace(start, stop, num_mult)

        # x,y -> x,x,x,x,y,y,y,y,x,x,x,x,y,y,y,y...
        tmp = np.tile(np.repeat(np.array(levels), step_size), time_reps)
        # get first num_step elements from row above
        # row -> row*m1; row*m2;row*m1;row*m2;
        tmp = np.tile(np.outer(scaling_range[:, np.newaxis], tmp[:num_steps]),
                      (photor_reps, 1))

        # get first num_photor_rows from array above
        return self.make_same(self.add_gwn(tmp[:num_photors, :])).T


def get_singleinput_cls(input_type):
    # I think this implementation will find only the classes defined in this
    # file
    all_input_cls = PhotorInput.__subclasses__()
    all_input_names = [cls.__name__ for cls in all_input_cls]
    try:
        input_cls = all_input_cls[all_input_names.index(input_type)]
    except ValueError:
        print('Invalid PhotorInput subclass name: {}'.format(input_type))
        print('Valid names: {}'.format(all_input_names))
        return None

    return input_cls


def main():
    pass


if __name__ == '__main__':
    main()

