#!/usr/bin/env python

import os, resource, sys
import argparse

import networkx as nx
import numpy as np

import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer

import retina.retina as ret
import retina.geometry.hexagon as hx
import gen_input as gi
from retina.input_generator import RetinaInputGenerator
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.configreader import ConfigReader
from retina.LPU import LPU

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


def get_master_id(i):
    return 'retina{}'.format(i)


def get_worker_id(i):
    return 'retina{}'.format(i)


# number of neurons of `j`th worker out of `worker_num`
# with `total_neurons` neurons overall
def get_worker_num_neurons(j, total_neurons, worker_num):
    num_neurons = (total_neurons-1) // worker_num + 1
    return min(num_neurons, total_neurons - j*num_neurons)


def add_master_LPU(config, i, retina, manager, generator):
    dt = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    input_filename = config['Retina']['input_file']
    output_filename = config['Retina']['output_file']
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    if generator is None:
        input_file = '{}{}{}.h5'.format(input_filename, i, suffix)
    else:
        input_file = None
    output_file = '{}{}{}.h5'.format(output_filename, i, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, i, suffix)

    G = retina.get_master_graph()
    nx.write_gexf(G, gexf_file)

    n_dict_ret, s_dict_ret = LPU.lpu_parser(gexf_file)
    master_id = get_master_id(i)
    modules = []

    manager.add(LPU, master_id, dt, n_dict_ret, s_dict_ret,
                input_file=input_file, output_file=output_file,
                device=i, debug=debug, time_sync=time_sync,
                modules=modules, input_generator=generator)


def add_worker_LPU(config, i, j, retina, manager):
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    dt = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    eye_num = config['General']['eye_num']
    worker_num = config['Retina']['worker_num']
    gexf_file = '{}{}_{}{}.gexf.gz'.format(gexf_filename, i, j, suffix)

    G = retina.get_worker_graph(j+1, worker_num)
    G = nx.convert_node_labels_to_integers(G)
    nx.write_gexf(G, gexf_file)

    worker_dev = j + i*worker_num + eye_num

    print('Device worker number {}'.format(worker_dev))

    n_dict_ret, s_dict_ret = LPU.lpu_parser(gexf_file)
    worker_id = get_worker_id(worker_dev)
    modules = []
    manager.add(LPU, worker_id, dt, n_dict_ret, s_dict_ret,
                input_file=None, output_file=None,
                device=worker_dev, debug=debug, time_sync=time_sync,
                modules=modules, input_generator=None)


def connect_master_worker(config, i, j, retina, manager):
    total_neurons = retina.num_photoreceptors

    eye_num = config['General']['eye_num']
    worker_num = config['Retina']['worker_num']

    worker_dev = j + i*worker_num + eye_num

    master_id = get_master_id(i)
    worker_id = get_worker_id(worker_dev)
    print('Connecting {} and {}'.format(master_id, worker_id))

    with Timer('update of connections in Pattern object'):
        pattern = retina.update_pattern_master_worker(j+1, worker_num)

    with Timer('update of connections in Manager'):
        manager.connect(master_id, worker_id, pattern)


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

def get_input_gen(config):
    inputmethod = config['Retina']['inputmethod']

    if inputmethod == 'read':
        print('Generating input files')
        with Timer('input generation'):
            gi.gen_input(config)
        return None
    else:
        print('Using input generating function')

        return RetinaInputGenerator(config)


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

    eye_num = config['General']['eye_num']
    worker_num = config['Retina']['worker_num']
    num_rings = config['Retina']['rings']
    radius = config['Retina']['radius']
    eulerangles = config['Retina']['eulerangles']

    generator = get_input_gen(config)

    manager = core.Manager()
    for i in range(eye_num):
        with Timer('instantiation of retina #{}'.format(i)):
            transform = AlbersProjectionMap(radius,
                                            eulerangles[3*i:3*(i+1)]).invmap
            hexagon = hx.HexagonArray(num_rings=num_rings, radius=radius,
                                      transform=transform)

            retina = ret.RetinaArray(hexagon, config)

            # sets retina attribute which is required for the generation of
            # receptive fields
            if generator is not None:
                generator.generate_datafiles(i, retina)

            add_master_LPU(config, i, retina, manager, generator)

            for j in range(worker_num):
                add_worker_LPU(config, i, j, retina, manager)
                connect_master_worker(config, i, j, retina, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()
