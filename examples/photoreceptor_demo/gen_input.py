from __future__ import division

import neurokernel.LPU.utils.simpleio as sio
from neurokernel.tools.timing import Timer

from retina.input.photoreceptor_input import get_singleinput_cls


def gen_input_norfscreen(config):
    steps = config['General']['steps']
    input_file = config['Photoreceptor']['input_file']
    input_type = config['Photoreceptor']['single_intype']
    input_obj = get_singleinput_cls(input_type)(config)

    with Timer('getting photoreceptor inputs'):
        # although not configurable one can use
        # alternatives like .get_flat_image()
        # if available by manually
        # changing command below
        photor_inputs = input_obj.get_input(steps)

        config['General']['steps'] = photor_inputs[:, 0].size

    sio.write_array(photor_inputs, filename=input_file)

