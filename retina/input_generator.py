
import numpy as np

import h5py

from neurokernel.LPU.utils.simpleio import *

import classmapper as cls_map


class RetinaInputGenerator(object):
    def __init__(self, config):
        self.config = config

        self.screen_type = config['Retina']['screentype']
        self.filtermethod = config['Retina']['filtermethod']
        screen_cls = cls_map.get_screen_cls(self.screen_type)
        self.screen = screen_cls(config)

    def generate_datafiles(self, i, retina):
        self.retina = retina

        screen = self.screen
        
        screen.setup_file('intensities{}.h5'.format(i))

        retina_elev_file = 'retina_elev{}.h5'.format(i)
        retina_azim_file = 'retina_azim{}.h5'.format(i)

        screen_dima_file = 'grid_dima{}.h5'.format(i)
        screen_dimb_file = 'grid_dimb{}.h5'.format(i)

        retina_dima_file = 'retina_dima{}.h5'.format(i)
        retina_dimb_file = 'retina_dimb{}.h5'.format(i)
        
        self.input_file = 'retina_input{}.h5'.format(i)

        elev_v, azim_v = retina.get_ommatidia_pos()
        rfs = self.generate_receptive_fields_no_gpu()

        for data, filename in [(elev_v, retina_elev_file),
                               (azim_v, retina_azim_file),
                               (screen.grid[0], screen_dima_file),
                               (screen.grid[1], screen_dimb_file),
                               (rfs.refa, retina_dima_file),
                               (rfs.refb, retina_dimb_file)]:
            write_array(data, filename)
    
        self.file_open = False

    def generate_receptive_fields_no_gpu(self):
        #TODO intensities file should also be written but is omitted for
        # performance reasons
        retina = self.retina
        screen = self.screen
        screen_type = self.screen_type

        mapdr_cls = cls_map.get_mapdr_cls(screen_type)
        projection_map = mapdr_cls.from_retina_screen(retina, screen)

        rf_params = projection_map.map(*retina.get_all_photoreceptors_dir())
        if np.isnan(np.sum(rf_params)):
            print('Warning, Nan entry in array of receptive field centers')

        vrf_cls = cls_map.get_vrf_no_gpu_cls(screen_type)
        rfs = vrf_cls(screen.grid)
        rfs.load_parameters(refa=rf_params[0], refb=rf_params[1],
                            acceptance_angle=retina.get_angle(),
                            radius=screen.radius)

        return rfs

    def generate_receptive_fields(self):
        #TODO intensities file should also be written but is omitted for
        # performance reasons
        retina = self.retina
        screen = self.screen
        screen_type = self.screen_type
        filtermethod = self.filtermethod

        mapdr_cls = cls_map.get_mapdr_cls(screen_type)
        projection_map = mapdr_cls.from_retina_screen(retina, screen)

        rf_params = projection_map.map(*retina.get_all_photoreceptors_dir())
        if np.isnan(np.sum(rf_params)):
            print('Warning, Nan entry in array of receptive field centers')

        if filtermethod == 'gpu':
            vrf_cls = cls_map.get_vrf_cls(screen_type)
        else:
            vrf_cls = cls_map.get_vrf_no_gpu_cls(screen_type)
        rfs = vrf_cls(screen.grid)
        rfs.load_parameters(refa=rf_params[0], refb=rf_params[1],
                            acceptance_angle=retina.get_angle(),
                            radius=screen.radius)

        self.rfs = rfs

    def next_input(self):
        if not self.file_open:
            self.input_file_handle = h5py.File(self.input_file, 'w')
            self.input_file_handle.create_dataset(
                        '/array',
                        (0, self.retina.num_photoreceptors),
                        dtype=np.float64,
                        maxshape=(None, self.retina.num_photoreceptors))
            self.file_open = True
        im = self.screen.get_screen_intensity_steps(1)
        inputs = self.rfs.filter_image(im)
        dataset_append(self.input_file_handle['/array'],
                       inputs.get().reshape((1, -1)))
        return inputs

    def __del__(self):
        try:
            self.input_file_handle.close()
        except:
            pass
