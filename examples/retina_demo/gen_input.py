from __future__ import division

import argparse
import atexit

import numpy as np
import pycuda.driver as cuda

import neurokernel.LPU.utils.simpleio as sio

import retina.retina as ret
import retina.geometry.hexagon as hx
import retina.classmapper as cls_map
from retina.screen.map.mapimpl import AlbersProjectionMap


def gen_input(config):
    cuda.init()
    ctx = cuda.Device(0).make_context()
    atexit.register(ctx.pop)

    suffix = config['General']['file_suffix']

    eye_num = 1 # config['General']['eye_num']

    eulerangles = config['Retina']['eulerangles']
    radius = config['Retina']['radius']

    rings = config['Retina']['rings']
    steps = config['General']['steps']

    screen_type = config['Retina']['screentype']
    screen_cls = cls_map.get_screen_cls(screen_type)

    for i in range(eye_num):
        screen = screen_cls(config)
        screen.setup_file('intensities{}{}.h5'.format(suffix,i))
    
        retina_elev_file = 'retina_elev{}.h5'.format(i)
        retina_azim_file = 'retina_azim{}.h5'.format(i)

        screen_dima_file = 'grid_dima{}.h5'.format(i)
        screen_dimb_file = 'grid_dimb{}.h5'.format(i)

        retina_dima_file = 'retina_dima{}.h5'.format(i)
        retina_dimb_file = 'retina_dimb{}.h5'.format(i)

        input_file = 'retina_input{}.h5'.format(i)

        transform = AlbersProjectionMap(radius,
                                        eulerangles[3*i:3*(i+1)]).invmap
        hexagon = hx.HexagonArray(num_rings=rings, radius=radius,
                                  transform=transform)
        retina = ret.RetinaArray(hexagon, config)
        print('Acceptance angle: {}'.format(retina.acceptance_angle))
        print('Neurons: {}'.format(retina.num_photoreceptors))

        elev_v, azim_v = retina.get_ommatidia_pos()

        rfs = _get_receptive_fields(retina, screen, screen_type)
        steps_count = steps
        write_mode = 'w'
        while (steps_count > 0):
            steps_batch = min(100, steps_count)
            im = screen.get_screen_intensity_steps(steps_batch)
            photor_inputs = rfs.filter(im)
            sio.write_array(photor_inputs, filename=input_file, mode=write_mode)
            steps_count -= steps_batch
            write_mode = 'a'

        for data, filename in [(elev_v, retina_elev_file),
                               (azim_v, retina_azim_file),
                               (screen.grid[0], screen_dima_file),
                               (screen.grid[1], screen_dimb_file),
                               (rfs.refa, retina_dima_file),
                               (rfs.refb, retina_dimb_file)]:
            sio.write_array(data, filename)

def _get_receptive_fields(retina, screen, screen_type):
    mapdr_cls = cls_map.get_mapdr_cls(screen_type)
    projection_map = mapdr_cls.from_retina_screen(retina, screen)

    rf_params = projection_map.map(*retina.get_all_photoreceptors_dir())
    if np.isnan(np.sum(rf_params)):
        print('Warning, Nan entry in array of receptive field centers')
    vrf_cls = cls_map.get_vrf_cls(screen_type)
    rfs = vrf_cls(screen.grid)
    rfs.load_parameters(refa=rf_params[0], refb=rf_params[1],
                        acceptance_angle=retina.get_angle(),
                        radius=screen.radius)
    return rfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rings', type=int,
                        default=14,
                        help='number of layers of ommatidia on circle')
    parser.add_argument('-l', dest='sublpus', type=int,
                        default=0,
                        help='number of sublpus')
    parser.add_argument('--steps', default=1000, type=int,
                        help='simulation steps')
    args = parser.parse_args()

    gen_input(args.steps, args.rings, 1, args.sublpus)


if __name__ == '__main__':
    main()

