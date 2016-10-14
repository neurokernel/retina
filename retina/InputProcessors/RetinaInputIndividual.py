#!/usr/bin/env python

import numpy as np
import h5py
from neurokernel.LPU.utils.simpleio import *

import retina.classmapper as cls_map
import pycuda.driver as cuda

from neurokernel.LPU.InputProcessors.BaseInputProcessor import BaseInputProcessor

class RetinaInputIndividual(BaseInputProcessor):
    def __init__(self, config, photoreceptor_list):
        """
        config: see retina configuration template
        
        photoreceptor_list: a list of tuple get from
                            networkx.MultiDigraph.nodes(data=True),
                            i.e., the first element of the tuple is the id
                            and the second is a dictionary containing parameters
                            of the photoreceptor
        """
        self.config = config

        self.screen_type = config['Retina']['screentype']
        self.filtermethod = config['Retina']['filtermethod']
        screen_cls = cls_map.get_screen_cls(self.screen_type)
        self.screen = screen_cls(config)
        #self.retina = retina
        self.pr_list = photoreceptor_list
        self.retina_radius = config['Retina']['radius']
        self.num_photoreceptors = len(photoreceptor_list)
        
        #uids = ['neuron_{}_{}'.format(name, i) for i in range(retina.num_elements)
        #        for name in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']]
        uids = [a[0] for a in self.pr_list]
        
        super(RetinaInputIndividual, self).__init__([('photon',uids)], mode=0)

    def pre_run(self):
        self.generate_receptive_fields()
        self.generate_datafiles()
        self.input_file_handle = h5py.File(self.input_file, 'w')
        self.input_file_handle.create_dataset(
                    '/array',
                    (0, self.num_photoreceptors),
                    dtype=np.float64,
                    maxshape=(None, self.num_photoreceptors))
    
    def generate_datafiles(self):
        screen = self.screen
        #retina = self.retina
        pr_list = self.pr_list
        rfs = self.rfs
        i = 0
        
        screen.setup_file('intensities{}.h5'.format(i))

        retina_elev_file = 'retina_elev{}.h5'.format(i)
        retina_azim_file = 'retina_azim{}.h5'.format(i)

        screen_dima_file = 'grid_dima{}.h5'.format(i)
        screen_dimb_file = 'grid_dimb{}.h5'.format(i)

        retina_dima_file = 'retina_dima{}.h5'.format(i)
        retina_dimb_file = 'retina_dimb{}.h5'.format(i)
        
        self.input_file = 'retina_input{}.h5'.format(i)

        #elev_v, azim_v = retina.get_ommatidia_pos()
        elev_v = np.array([a[1]['3d_elev'] for a in pr_list])
        azim_v = np.array([a[1]['3d_azim'] for a in pr_list])

        for data, filename in [(elev_v, retina_elev_file),
                               (azim_v, retina_azim_file),
                               (screen.grid[0], screen_dima_file),
                               (screen.grid[1], screen_dimb_file),
                               (rfs.refa, retina_dima_file),
                               (rfs.refb, retina_dimb_file)]:
            write_array(data, filename)
    
        self.file_open = False

    def generate_receptive_fields(self):
        #TODO intensities file should also be written but is omitted for
        # performance reasons
        
        #retina = self.retina
        pr_list = self.pr_list
        screen = self.screen
        screen_type = self.screen_type
        filtermethod = self.filtermethod

        mapdr_cls = cls_map.get_mapdr_cls(screen_type)
        projection_map = mapdr_cls(self.retina_radius, screen.radius)
        
        pos_elev = np.array([a[1]['3d_elev'] for a in pr_list])
        pos_azim = np.array([a[1]['3d_azim'] for a in pr_list])
        dir_elev = np.array([a[1]['optic_axis_elev'] for a in pr_list])
        dir_azim = np.array([a[1]['optic_axis_azim'] for a in pr_list])

        rf_params = projection_map.map(pos_elev, pos_azim, dir_elev, dir_azim)
        if np.isnan(np.sum(rf_params)):
            print('Warning, Nan entry in array of receptive field centers')

        if filtermethod == 'gpu':
            vrf_cls = cls_map.get_vrf_cls(screen_type)
        else:
            vrf_cls = cls_map.get_vrf_no_gpu_cls(screen_type)
        rfs = vrf_cls(screen.grid)
        rfs.load_parameters(refa=rf_params[0], refb=rf_params[1],
                            acceptance_angle=pr_list[0][1]['acceptance_angle'],
                            radius=screen.radius)

        rfs.generate_filters()
        self.rfs = rfs
    
    def update_input(self):
        im = self.screen.get_screen_intensity_steps(1)
        # reshape neede for inputs in order to write file to an array
        inputs = self.rfs.filter_image_use(im).get().reshape((1,-1))
        dataset_append(self.input_file_handle['/array'],
                       inputs)
        self.variables['photon']['input'][:] = inputs
                         
    
    def is_input_available(self):
        return True
    
    def post_run(self):
        self.input_file_handle.close()


    def __del__(self):
        try:
            self.input_file_handle.close()
        except:
            pass
    
