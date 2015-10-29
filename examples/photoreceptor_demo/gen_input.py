from __future__ import division

import argparse
import atexit

import numpy as np
import pycuda.driver as cuda

import neurokernel.LPU.utils.simpleio as sio
from neurokernel.tools.timing import Timer

import retina.retina as ret
import retina.geometry.hexagon as hx
import retina.classmapper as cls_map

from retina.input.photoreceptor_input import get_singleinput_cls

DEMO_1 = 'master'
DEMO_2 = 'distributed'


def gen_input_norfscreen(config):
    steps = config['General']['steps']
    input_file = config['Photoreceptor']['input_file']
    input_type = config['Photoreceptor']['single_intype']
    input_obj = get_singleinput_cls(input_type)(config)

    with Timer('getting photoreceptor inputs'):
        # although not configurable one can use
        # alternatives like .get_flat_image()
        # or .get_phase_gwn if available by
        # manually changing command below
        photor_inputs = input_obj.get_input(steps)

        config['General']['steps'] = photor_inputs[:, 0].size

    sio.write_array(photor_inputs, filename=input_file)


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

