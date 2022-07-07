#!/usr/bin/env python

import os, resource, sys
import argparse

import networkx as nx
import numpy as np

import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU

import retina.retina as ret
import retina.geometry.hexagon as hx

from retina.InputProcessors.RetinaInputProcessor import RetinaInputProcessor
from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.PhotoreceptorModel import PhotoreceptorModel
from retina.NDComponents.MembraneModels.BufferPhoton import BufferPhoton

dtype = np.double
RECURSION_LIMIT = 80000


def setup_logging(config):
    log = config['General']['log']
    file_name = None
    screen = False

    if log in ['file', 'both']:
        file_name = 'neurokernel.log'
    if log in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)

def get_retina_id(i):
    return 'retina{}'.format(i)

# number of neurons of `j`th worker out of `worker_num`
# with `total_neurons` neurons overall
def get_worker_num_neurons(j, total_neurons, worker_num):
    num_neurons = (total_neurons-1) // worker_num + 1
    return min(num_neurons, total_neurons - j*num_neurons)


def add_retina_LPU(config, i, retina, manager):
    T = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    input_filename = config['Retina']['input_file']
    output_filename = config['Retina']['output_file']
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    output_file = '{}{}{}.h5'.format(output_filename, i, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, i, suffix)

    inputmethod = config['Retina']['inputmethod']
    if inputmethod == 'read':
       
        print('Reading from previously generated input')
        print('The configuration must stay the same.')
        input_processor = FileInputProcessor('{}.h5'.format(config['Retina']['input_file']))
    else:
        print('Using input generating function')
        input_processor = RetinaInputProcessor(config, retina)
        print(input_processor.for_test)

    output_processor = FileOutputProcessor([('V',None)], output_file, sample_interval=1)

    G = retina.get_worker_nomaster_graph()
    nx.write_gexf(G, gexf_file)

    (comp_dict, conns) = LPU.graph_to_dicts(G)
    retina_id = get_retina_id(i)

    extra_comps = [PhotoreceptorModel, BufferPhoton]

    manager.add(LPU, retina_id, T, comp_dict, conns,
                device = i, input_processors = [input_processor],
                output_processors = [output_processor],
                debug=debug, time_sync=time_sync, extra_comps = extra_comps)


def start_simulation(config, manager):
    steps = config['General']['steps']
    with Timer('retina simulation'):
        manager.spawn()
        print('Manager spawned')
        manager.start(steps=steps)
        manager.wait()


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
        values = range(1, 5)

        index %= len(values)
        config['General']['file_suffix'] = suffixes[index]
        config['Retina']['worker_num'] = values[index]


def get_config_obj(args):
    conf_name = args.config

    # append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    conf_specname = os.path.join('..', 'template_spec.cfg')

    return ConfigReader(conf_filename, conf_specname)


def main():
    import neurokernel.mpi_relaunch
    # default limit is low for pickling
    # the data structures passed through mpi
    sys.setrecursionlimit(RECURSION_LIMIT)
    resource.setrlimit(resource.RLIMIT_STACK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default',
                        help='configuration file')
    parser.add_argument('-v', '--value', type=int, default=-1,
                        help='Value that can overwrite configuration '
                             'by changing this script accordingly. '
                             'It is useful when need to run this script '
                             'repeatedly for different configuration')

    args = parser.parse_args()

    with Timer('getting configuration'):
        conf_obj = get_config_obj(args)
        config = conf_obj.conf
        change_config(config, args.value)

    setup_logging(config)

    num_rings = config['Retina']['rings']
    radius = config['Retina']['radius']
    eulerangles = config['Retina']['eulerangles']

    manager = core.Manager()

    with Timer('instantiation of retina'):
        transform = AlbersProjectionMap(radius, eulerangles).invmap
        hexagon = hx.HexagonArray(num_rings=num_rings, radius=radius,
                                  transform=transform)

        retina = ret.RetinaArray(hexagon, config)

        # sets retina attribute which is required for the generation of
        # receptive fields
        #if generator is not None:
        #    generator.generate_datafiles(0, retina)

        add_retina_LPU(config, 0, retina, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()
