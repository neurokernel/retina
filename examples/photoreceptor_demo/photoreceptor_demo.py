#!/usr/bin/env python
from __future__ import division

import os
import argparse

import numpy as np
import networkx as nx

import neurokernel.core_gpu as core
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.PhotoreceptorModel import PhotoreceptorModel


dtype = np.double


def setup_logging(config):
    log = config['General']['log']
    file_name = None
    screen = False

    if log in ['file', 'both']:
        file_name = 'neurokernel.log'
    if log in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)


def generate_gexf(config, output_file):
    G = nx.DiGraph()

    micro = config['micro']
    photoreceptors = config['photoreceptors']

    for i in range(photoreceptors):
        G.node[i] = {
            'class': 'PhotoreceptorModel',
            'name': 'photoreceptor{}'.format(i),
            'selector': '/photor{}'.format(i),
            'num_microvilli': micro,
            'init_V': -81.99, 'init_sa': 0.2184, 'init_si': 0.9653,
            'init_dra': 0.0117, 'init_dri': 0.9998, 'init_nov': 0.0017
        }

    return G

def add_LPU(config, manager):
    config_photor = config['Photoreceptor']
    gexf_file = config_photor['gexf_file']
    input_file = config_photor['input_file']
    output_file = config_photor['output_file']

    G = generate_gexf(config_photor, gexf_file)

    LPU.graph_to_dicts(G)
    comp_dict, conns = LPU.graph_to_dicts(G)
    LPU_id = 'photoreceptor'
    debug = config_photor['debug']

    dt = config['General']['dt']
    extra_comps = [PhotoreceptorModel]

    input_processor = StepInputProcessor('photon', G.nodes(), 10000, 0.2, 1)
    output_processor = FileOutputProcessor([('V', G.nodes())], output_file, sample_interval=1)
    manager.add(LPU, LPU_id, dt, comp_dict, conns,
                device = 0, input_processors = [input_processor],
                output_processors = [output_processor],
                debug=debug, time_sync=False, extra_comps = extra_comps)


def start_simulation(config, manager):
    steps = config['General']['steps']
    with Timer('photoreceptor simulation'):
        manager.spawn()
        manager.start(steps=steps)
        manager.wait()


def add_suffix(filename, suffix):
    '''
        'name.gexf.gz' should become
        'namesuffix.gexf.gz'
    '''
    noext, exts = filename.split(os.extsep, 1)
    return '{noext}{suffix}{sep}{exts}'.format(noext=noext,
        suffix=suffix, sep=os.extsep, exts=exts)


def edit_files(config):
    '''
        Change filenames to add a suffix
    '''
    config_photor = config['Photoreceptor']
    suffix = config['General']['file_suffix']

    for file_type in ['gexf_file', 'input_file', 'output_file']:
        config_photor[file_type] = add_suffix(config_photor[file_type], suffix)


def change_config(config, index):
    '''
        Useful if one wants to run the same simulation
        with a few parameters changing based on index value

        Need to modify else part

        Parameters
        ----------
        config: configuration object
        index: simulation index
    '''
    if index < 0:
        pass
    else:
        suffixes = ['__{}'.format(i) for i in range(4)]
        values = 1000*range(1, 5)

        index %= len(values)
        config['General']['file_suffix'] = suffixes[index]
        config['General']['steps'] = values[index]


def get_config_obj(args):
    conf_name = args.config
    # append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    conf_specname = os.path.join('..', 'template_spec.cfg')

    return ConfigReader(conf_filename, conf_specname)


def main():
    import neurokernel.mpi_relaunch

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default',
                        help='configuration file')

    parser.add_argument('-v', '--value', type=int, default=-1,
                        help='Value that can overwrite configuration'
                             'by changing this script accordingly.'
                             'It is useful when need to run this script'
                             'repeatedly for different configuration')

    args = parser.parse_args()

    with Timer('getting configuration'):
        conf_obj = get_config_obj(args)
        config = conf_obj.conf
        change_config(config, args.value)
        edit_files(config)

    setup_logging(config)
    # with Timer('input generation'):
    #     gi.gen_input_norfscreen(config)

    manager = core.Manager()
    with Timer('photoreceptor instantiation'):
        add_LPU(config, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()
