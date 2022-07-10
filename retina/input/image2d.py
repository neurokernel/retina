from __future__ import division

import os
from abc import ABCMeta, abstractmethod

import numpy as np
from neurokernel.LPU.utils.simpleio import *


class Image2D(object):
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.dt = config['General']['dt']
        self.dtype = np.double

        # number of 'pixels' in each dimension
        self.shape = tuple(config['InputType']['shape'])
        #self.shape = tuple([self.shape[1], self.shape[0]])
        self.infilename = config['InputType']['infilename']
        self.writefile = config['InputType']['writefile']

    def get_grid(self, xmin, xmax, ymin, ymax):
        print(f'the size of origin grid is x{self.shape[0]}, y{self.shape[1]}')
        return [np.linspace(xmin, xmax, self.shape[0]),
                np.linspace(ymin, ymax, self.shape[1])]

    def generate_2dimage(self, num_steps):
        #im_v = np.empty((num_steps,) + self.shape, dtype=self.dtype)
        im_v = np.empty((num_steps,self.shape[1],self.shape[0]), dtype=self.dtype)
        for i in range(num_steps):
            im_v[i] = self._generate_2dimage_step(self._internal_step)
            self._internal_step += 1

        if self.writefile:
            # for multiple calls file opening and closing has to be split
            write_array(im_v, self.infilename, complevel=9)

        return im_v

    # only resets internal step for now
    def reset(self):
        self._internal_step = 0

    @abstractmethod
    def _generate_2dimage_step(self, step):
        pass

    def get_input_config(self, config):
        intype = config['Retina']['intype']
        return config['InputType'][intype]


class BaseImage():

    baseparams = ['speed', 'levels']

    def __init__(self, **kwargs):
        self.set_base_parameters(**kwargs)

    def set_base_parameters(self, config):
        self.speed = config['speed']
        self.levels = tuple(config['levels'])


# Classes in alphabetic order
class Ball(Image2D, BaseImage):
    def __init__(self, config):
        Image2D.__init__(self, config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.set_base_parameters(config)
        self.white_back = config['white_back']

        rel_position = config['center']
        shape = self.shape

        # add more options later if needed
        if rel_position == 'center':
            self.center = (shape[1]/2, shape[0]/2)
        else:
            # default center at the center of the image
            self.center = (shape[1]/2, shape[0]/2)

        self.x, self.y = np.meshgrid(np.arange(float(shape[1])),
                                     np.arange(float(shape[0])))
        self.reset()

    def _generate_2dimage_step(self, step):
        dt = self.dt
        levels = self.levels
        center = self.center
        speed = self.speed
        white_back = self.white_back

        im = (1 if white_back else -1) * np.sign(
            np.sqrt((self.x-center[0])**2+(self.y-center[1])**2)
            - step*dt*speed).astype(np.double)

        return im*((levels[1]-levels[0])/2) + (levels[1]+levels[0])/2


class Bar(Image2D, BaseImage):
    def __init__(self, config):
        Image2D.__init__(self, config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.set_base_parameters(config)
        self.dir = config['direction']
        self.bar_width = config['bar_width']
        self.double = config['double']
        self.reset()

    def _generate_2dimage_step(self, step):
        shape = tuple([self.shape[1], self.shape[0]])

        im = np.ones(shape, dtype=self.dtype)*self.levels[0]
        if self.dir == 'v':  # vertical movement
            st1 = int(np.mod(step*self.speed*self.dt, shape[0]))
            en1 = min(int(st1 + self.bar_width), shape[0])
            im[st1:en1, :] = self.levels[1]
            if self.double:
                st2 = min(int(en1 + self.bar_width), shape[0])
                en2 = min(int(st2 + self.bar_width), shape[0])
                im[st2:en2, :] = self.levels[1]
        elif self.dir == 'h':  # horizontal movement
            st1 = int(np.mod(step*self.speed*self.dt, shape[1]))
            en1 = min(int(st1 + self.bar_width), shape[1])
            im[:, st1:en1] = self.levels[1]
            if self.double:
                st2 = min(int(en1 + self.bar_width), shape[1])
                en2 = min(int(st2 + self.bar_width), shape[1])
                im[:, st2:en2] = self.levels[1]
        else:
            raise ValueError('Invalid value for direction {}'
                             .format(self.dir))
        return im


class FlickerStep(Image2D):
    def __init__(self, config):
        super(FlickerStep, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.frequency = config['frequency']
        self.levels = tuple(config['levels'])
        self.count = 0
        self.reset()

    def _generate_2dimage_step(self, step):
        shape = self.shape

        im = np.ones(shape, dtype=self.dtype)*self.levels[self.count]
        if step >= int(1./self.frequency/2/self.dt):
            self.reset()
            self.count = (self.count + 1) % len(self.levels)

        return im


class Gratings(Image2D):
    def __init__(self, config):
        super(Gratings, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        self.x_freq = config['x_freq']
        self.y_freq = config['y_freq']
        self.x_speed = config['x_speed']
        self.y_speed = config['y_speed']
        self.sinusoidal = config['sinusoidal']
        self.levels = tuple(config['levels'])
        self.reset()

    def _generate_2dimage_step(self, step):
        dt = self.dt
        levels = self.levels
        shape = self.shape
        x_freq = self.x_freq
        y_freq = self.y_freq
        x_speed = self.x_speed
        y_speed = self.y_speed
        sinusoidal = self.sinusoidal

        x, y = np.meshgrid(np.arange(float(shape[1])),
                           np.arange(float(shape[0])))

        if sinusoidal:
            sinfunc = lambda w: ((np.sin(w)+1)/2)*(levels[1]-levels[0]) \
                + levels[0]
        else:
            sinfunc = lambda w: np.sign(np.cos(w))*((levels[1]-levels[0])/2) \
                + (levels[1]+levels[0])/2

        return sinfunc(x_freq*2*np.pi*(x - x_speed*step*dt) +
                       y_freq*2*np.pi*(y - y_speed*step*dt))

    def get_config(self):
        return self.get_config_from_params(['x_freq', 'y_freq', 'x_speed',
                                            'y_speed', 'sinusoidal', 'levels'])


class Natural(Image2D):
    def __init__(self, config):
        super(Natural, self).__init__(config)

        input_config = self.get_input_config(config)
        self.set_parameters(input_config)

    def set_parameters(self, config):
        np.random.seed(config['seed'])
        self.store_coords = config['store_coords']
        self.coord_file_name = config['coord_file']
        self.image_file = config['image_file']
        self.scale = config['scale']
        self.speed = config['speed']
        self.image = self.readimage()
        self.image *= self.scale/self.image.max()

        # start from the middle of the image
        self.imagex = self.image.shape[0]/2
        self.imagey = self.image.shape[1]/2
        self.vx = np.random.randn(1)*self.speed
        self.vy = np.random.randn(1)*self.speed
        self.margin = 10
        self.file_open = False

        self.reset()

    def __del__(self):
        try:
            if self.store_coords:
                print('closing natural_xy file')
                self.coordfile.close()
        except AttributeError:
            pass

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

    def generate_2dimage(self, num_steps):
        # photons are stored with unit photons/sec
        dt = self.dt
        shape = self.shape
        #shape = tuple([self.shape[1], self.shape[0]])
        vx = self.vx
        vy = self.vy
        image = self.image
        imagex = self.imagex
        imagey = self.imagey
        margin = self.margin

        if not self.file_open:
            if self.store_coords:
                self.coordfile = h5py.File(self.coord_file_name, 'w')
                self.coordfile.create_dataset('/array', (0, 2), dtype=np.int32,
                                              maxshape = (None, 2))
            self.file_open = True

        xy = np.zeros((num_steps, 2), np.int32)

        h_im, w_im = image.shape

        im_v = np.empty((num_steps,self.shape[1],self.shape[0]), dtype=self.dtype)
        #im_v = np.empty((num_steps,)+self.shape, dtype=self.dtype)
        for i in range(num_steps):
            imagex = imagex + vx*dt
            imagey = imagey + vy*dt

            # change direction if bound is reached
            imagex_max = imagex + shape[0]
            if imagex_max + margin > h_im:
                imagex = h_im - margin - shape[0]
                vx = -vx

            if imagex - margin <= 0:
                imagex = margin
                vx = -vx

            imagey_max = imagey + shape[0]
            if imagey_max + margin > w_im:
                imagey = w_im - margin - shape[1]
                vy = -vy

            if imagey - margin <= 0:
                imagey = margin
                vy = -vy

            # decimal indices are allowed and decimal part is ignored
            im_v[i] = image[int(np.around(imagex)):int(np.around(imagex)) + self.shape[1],
                            int(np.around(imagey)):int(np.around(imagey)) + self.shape[0]]
            xy[i, :] = (imagex, imagey)  # convert to int

            # change speed/direction every about 1/dt steps
            if np.random.rand() < dt:
                vx = np.random.randn(1)*self.speed

            if np.random.rand() < dt:
                vy = np.random.randn(1)*self.speed

        self.imagex = imagex
        self.imagey = imagey
        self.vx = vx
        self.vy = vy
        if self.store_coords:
            dataset_append(self.coordfile['/array'], xy)

        if self.writefile:
            # for multiple calls file opening and closing has to be split
            write_array(im_v, self.infilename, complevel=9)

        #return im_v.swapaxes(1, 2)
        im_v = im_v[:, :, ::-1]
        return im_v

    # Not used, class overrides generate_2dimage
    # and this function is not supposed to be called
    def _generate_2dimage_step(self, step):
        pass


class Video(Image2D):
    def __init__(self, config):
        super(Video, self).__init__(config)
        input_config = self.get_input_config(config)
        self.set_parameters(input_config)
        

    def set_parameters(self, config):

        self.video_file = config['video_file']
        self.steps = config['steps']
        self.scale = config['scale']
        self.video_array = self.load_video()
        self.shape = tuple([np.shape(self.video_array)[2], np.shape(self.video_array)[1]])
        self.retina_frames = self.adapt_frames()
        #self.retina_input_video = self.adapt_video()
        #self.retina_input_video *= self.scale/np.max(self.retina_input_video)
        self.file_open = False
        self.reset()


    def load_video(self):
        from retina.input.video_reader import video_capture, video_adapter
        try:
            video_array = video_capture(self.video_file, self.scale)
        except AttributeError:
            print('Tried to read video before setting the file variable')
            raise
        except IOError:
            if self.video_file is None:
                print('Video file not specified')
            raise
  
        return video_array[:, :, ::-1]

    def adapt_frames(self):
        from retina.input.video_reader import video_capture, video_adapter, frames_adapter
        steps = self.steps
        dt = self.dt
        retina_input_frames = frames_adapter(self.video_array, dt, steps)
        return retina_input_frames
    
    def adapt_video(self):
        # may use too much memory, stop using this one
        from retina.input.video_reader import video_capture, video_adapter
        #steps = config['General']['steps']
        steps = self.steps
        dt = self.dt
        retina_input_video, retina_input_video_info = video_adapter(self.video_array, dt, steps)
        retina_input_video = retina_input_video[:, :, ::-1]
        return retina_input_video
        

    def _generate_2dimage_step(self, step):
        frame_index_now = int(self.retina_frames[step])
        frame_now = self.video_array[frame_index_now]
        return frame_now
        #return self.retina_input_video[step]


def image2Dfactory(input_type):
    # I think this implementation will find only the classes defined in this file
    # I think so, but isn't it enough?
    all_image2d_cls = Image2D.__subclasses__()
    all_image2d_names = [cls.__name__ for cls in all_image2d_cls]
    try:
        image2d_cls = all_image2d_cls[all_image2d_names.index(input_type)]
    except ValueError:
        print('Invalid Image2D subclass name: {}'.format(input_type))
        print('Valid names: {}'.format(all_image2d_names))
        return None

    return image2d_cls


def savefig(values, ylabel, imagefile):
    '''
        Generates a graph from the given values and saves it in a file
    '''
    import pylab
    import matplotlib.pyplot as plt

    plt.hold(False)
    plt.plot(values)
    plt.ylabel(ylabel)

    pylab.savefig(imagefile)


def savemp4(images, videofile, step=10):
    '''
        Generates a frame every 10(default) images and saves all
        of them to a video file

        parameters:
            images: a numpy array where each row corresponds to an image
            videofile: file to store video
            step: every that number of images will be stored in the file
                  the rest will be ignored (e.g if step is 10
                  and images are 50, only images 1,11,21,31,41 will be stored)
    '''
    from matplotlib import cm
    from matplotlib.animation import FFMpegFileWriter, AVConvFileWriter
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, MovieWriter
    import matplotlib.animation as animation
    
    # this is written with matplotlib 3.0, the old version may not work anymore
    images_cut = images[::step, :, :]
    def update_plot(frame_number, images_cut, plot):
        if frame_number%50 == 0:
            print(f'now is {frame_number}th frame')
        
        plot[0].remove()
        plot[0] = ax.imshow(images_cut[frame_number], 
                            cmap=cm.Greys_r, vmin=images_cut.min(), vmax=images_cut.max())
    
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)


    plot = [ax.imshow(images_cut[0], cmap=cm.Greys_r, vmin=images_cut.min(), vmax=images_cut.max())]
    ax.set_title('input before mapping into screen')

    fps = 10
    frn = len(images_cut)
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(images_cut, plot), interval=1000/fps)


    Writer = animation.writers['ffmpeg']
    ani.save('motoko.mp4', writer=Writer(fps=10))
    

def main():
    from neurokernel.tools.timing import Timer
    from retina.configreader import ConfigReader

    def test_image2D(imtype, write_video, write_image, write_screen, config):
        steps = config['General']['steps']
        config['General']['intype'] = imtype
        image2Dgen = image2Dfactory(imtype)(config)

        with Timer('generation of {} example input(steps {})'
                   .format(imtype, steps)):
            images = image2Dgen.generate_2dimage(num_steps=steps)

        if write_video:
            with Timer('writing video file'):
                savemp4(images, '{}_input.mp4'.format(imtype.lower()))

        if write_image:
            with Timer('writing image file'):
                savefig(np.log10(images[:, 1, 1]+1e-10),
                        'logarithm of {}'.format(imtype),
                        'temp_{}.png'.format(imtype))

        if write_screen:
            print('motoko!')
            print(type(images))
            print(np.shape(images))
            print('Images: \n{}'.format(images))

    def get_config():
        #conf_name = os.path.join('retina', 'retina', 'input',
        #                         'template.cfg')
        conf_name = os.path.join('retina_demo_moving_eye.cfg')
        conf_specname = os.path.join('template_spec.cfg')
        return ConfigReader(conf_name, conf_specname).conf
    
    np.set_printoptions(precision=3)

    #test_suite = ['Ball', 'Bar', 'FlickerStep', 'Natural']
    test_suite = ['Natural']
    
    write_video = True
    write_image = False
    write_screen = False
    config = get_config()

    for cls in test_suite:
        test_image2D(cls, write_video, write_image, write_screen, config)

''' run as `python -m retina.model.image2d` '''
if __name__ == '__main__':
    main()
