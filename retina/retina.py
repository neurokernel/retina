from __future__ import division

import collections
import importlib

import networkx as nx
import numpy as np
PI = np.pi

from neurokernel.pattern import Pattern

from geometry.opticaxis import opticaxisFactory, RuleHexArrayMap, OpticAxisRule

def divceil(x, y):
    return (x+y-1)//y


class Ommatidium(object):
    def __init__(self, element):
        '''
            element: ArrayElement object
        '''
        elev, azim = element.dima, element.dimb
        self._elev = elev
        self._azim = azim
        self.element = element

        # maybe simple dic is sufficient
        self.neurons = collections.OrderedDict()

    def add_neuron(self, neuron):
        if neuron.name not in self.neurons:
            self.neurons[neuron.name] = neuron

    def get_neighborid(self, neighbor_dr):
        return self.element.get_neighborid(neighbor_dr)

    @property
    def is_dummy(self):
        return self.element.is_dummy

    @property
    def sphere_pos(self):
        return self._elev, self._azim

    @property
    def gid(self):
        return self.element.gid

    @property
    def photoreceptor_num(self):
        return sum(n.is_photoreceptor
                   for n in self.neurons.values())

    @property
    def neuron_num(self):
        return len(self.neurons)


class RetinaArray(object):
    def __init__(self, hex_array, config):
        self.hex_array = hex_array

        modelname = config['Retina']['model']
        try:
            self.model = importlib.import_module('vision_models.{}'
                                                 .format(modelname))
        except ImportError:
            self.model = importlib.import_module('retina.vision_models.{}'
                                                 .format(modelname))

        self.opticaxis = opticaxisFactory('SuperpositionLT')()
        self.rulemap = RuleHexArrayMap(self.opticaxis, hex_array)

        self._set_elements()
        self._update_neurons()

        # in degrees
        self.interommatidial_angle = self._get_interommatidial_angle()

        # in degrees
        acc_factor = config['Retina']['acceptance_factor']
        self.acceptance_angle = self.interommatidial_angle * acc_factor

        self._generate_graph()

    def _set_elements(self):
        self._ommatidia = [Ommatidium(el)
                           for el in self.hex_array.elements]

    def _update_neurons(self):
        for omm in self._ommatidia:
            for neuron_params in self.model.OMMATIDIA_NEURON_LIST:
                self._add_photoreceptor(omm, neuron_params)

    def _add_photoreceptor(self, ommatidium, neuron_params):
        nid = self.opticaxis.name_to_ind(neuron_params['name'])
        neighbordr = self.opticaxis.neighbor_for_photor(nid)
        neighborid = ommatidium.get_neighborid(neighbordr)

        # position on sphere coincides with the
        # surface normal and the desired direction
        direction = self._ommatidia[neighborid].sphere_pos
        photor = OmmatidiumNeuron(ommatidium, direction, neuron_params)
        ommatidium.add_neuron(photor)

    def _add_fb_neuron(self, ommatidium, neuron_params):
        neuron = OmmatidiumNeuron(ommatidium, None, neuron_params,
                                  is_photoreceptor=False)
        ommatidium.add_neuron(neuron)

    def _get_interommatidial_angle(self):
        ''' Returns angle in degrees '''
        elev1, azim1 = self._ommatidia[0].sphere_pos
        try:
            elev2, azim2 = self._ommatidia[1].sphere_pos
        except IndexError:
            # when there is only one element
            # assume interommatidial angle is 90
            return 90

        x1 = np.sin(elev1)*np.cos(azim1)
        y1 = np.sin(elev1)*np.sin(azim1)
        z1 = np.cos(elev1)
        x2 = np.sin(elev2)*np.cos(azim2)
        y2 = np.sin(elev2)*np.sin(azim2)
        z2 = np.cos(elev2)
        angle = np.arccos(x1*x2 + y1*y2 + z1*z2)
        return float(angle*180/np.pi)

    def get_all_photoreceptors(self):
        return [photor for ommatidium in self._ommatidia
                for photor in ommatidium.neurons.itervalues()
                if photor.name in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']]

    def get_angle(self):
        return self.acceptance_angle

    def get_all_photoreceptors_dir(self):
        allphotors = self.get_all_photoreceptors()
        elevazim = np.array([photor.sphere_pos for photor in allphotors])
        dirs = np.array([photor.direction for photor in allphotors])

        return elevazim[:, 0], elevazim[:, 1], dirs[:, 0], dirs[:, 1]

    def get_ommatidia_pos(self):
        positions = np.array([omm.sphere_pos for omm in self._ommatidia])
        return positions[:, 0], positions[:, 1]

    def _generate_graph(self):
        G_master = nx.DiGraph()
        G_workers = nx.DiGraph()
        G_workers_nomaster = nx.DiGraph()

        num_photoreceptors = self.num_photoreceptors

        num_w1 = 0  # workers no master counter
        num_w2 = 0  # workers counter
        num_m = 0
        for i, omm in enumerate(self._ommatidia):
            for name, neuron in omm.neurons.items():
                neuron.num = num_w1
                G_workers_nomaster.add_node(num_w1, neuron.params.copy())
                G_workers_nomaster.node[num_w1].update(
                    {'selector': '/ret/{}/{}'.format(i, name)})
                num_w1 += 1

                if OpticAxisRule.is_photor(name):
                    ind = OpticAxisRule.name_to_ind(name)
                    G_workers.add_node(num_w2, neuron.params.copy())
                    G_workers.node[num_w2].update(
                        {'selector': '/ret/{}/R{}'.format(i, ind)})
                    G_workers.add_node(num_w2+num_photoreceptors, {
                        'class': 'Port',
                        'name': name,
                        'port_type': 'gpot',
                        'port_io': 'in',
                        'selector': '/ret/{}/in{}'.format(i, ind)
                    })
                    G_workers.add_edge(num_w2+num_photoreceptors,
                                       num_w2, type='directed')

                    num_w2 += 1

                    G_master.node[num_m] = {
                        'class': 'BufferNeuron',
                        'name': 'buf{}'.format(ind),
                        'selector': '/master/{}/buf{}'.format(i, ind)
                    }
                    G_master.node[num_m+num_photoreceptors] = {
                        'class': 'Port',
                        'port_type': 'gpot',
                        'port_io': 'in',
                        'name': name,
                        'selector': '/master/{}/R{}'.format(i, ind)
                    }
                    num_m += 1

#        num = 0
#        for omm in self._ommatidia:
#            for neuron in omm.neurons.itervalues():
#                for synapse in neuron.outgoing_synapses:
#                    synapse.params.update({'id': num})
#                    synapse.process_before_export()
#                    G_workers_nomaster.add_edge(synapse.pre_neuron.num,
#                               synapse.post_neuron.num,
#                               attr_dict=synapse.params.copy())
#                    num += 1

        self.G_master = G_master
        self.G_workers = G_workers
        self.G_workers_nomaster = G_workers_nomaster

    def get_worker_nomaster_graph(self, *args):
        try:
            return self.G_workers_nomaster.subgraph(self.get_nodes(*args))
        except TypeError:
            return self.G_workers_nomaster

    def get_worker_graph(self, *args):
        try:
            nodes = self.get_worker_nodes(*args)
            nodes += [i + self.num_photoreceptors for i in nodes]
            return self.G_workers.subgraph(nodes)
        except TypeError:
            return self.G_workers

    def get_master_graph(self):
        return self.G_master

    def update_pattern_master_worker(self, j, worker_num):
        indexes = self.get_worker_nodes(j, worker_num)
        
        master_selectors = self.get_master_selectors()
        worker_selectors = self.get_worker_selectors(j, worker_num)
        
        from_list = []
        to_list = []

        for i, ind in enumerate(indexes):
            col_m = ind // 6
            ind_m = 1 + (ind % 6)
            src = '/master/{}/buf{}'.format(col_m, ind_m)
            dest = '/ret/{}/in{}'.format(col_m, ind_m)
            from_list.append(src)
            to_list.append(dest)
            
            src = '/ret/{}/R{}'.format(col_m, ind_m)
            dest = '/master/{}/R{}'.format(col_m, ind_m)
            
            from_list.append(src)
            to_list.append(dest)

        pattern = Pattern.from_concat(','.join(master_selectors),
                                      ','.join(worker_selectors),
                                      from_sel = ','.join(from_list),
                                      to_sel = ','.join(to_list),
                                      gpot_sel = ','.join(from_list+to_list))
        return pattern

    # Neuron representation
    def get_neurons(self, j, sublpu_num):
        # numbering starts from 1
        ommatidia = self._ommatidia
        ommatidia_num = len(ommatidia)
        start = divceil((j-1)*ommatidia_num, sublpu_num)
        end = divceil(j*ommatidia_num, sublpu_num)
        neurons = []
        # implicit assumption i is the same as
        # the global id of
        for i in range(start, end):
            ommatidium = ommatidia[i]
            for omm_neuron in ommatidium.neurons.itervalues():
                neurons.append((ommatidium.gid,
                                omm_neuron.name,
                                ommatidium.sphere_pos))
        return neurons

    # Numeric order representation
    def get_interval(self, j, sublpu_num):
        '''
            Returns jth interval out of `sublpu_num`.
            Numbering starts from 1.
            Outputs are multiple of neuron_types.
            End index of last interval might be greater
            but not less than the number of photoreceptors.
            Indexes match positions returned by
            `get_all_photoreceptors`
        '''
        ommatidia_num = self.num_elements
        start = self.neuron_types*divceil((j-1)*ommatidia_num, sublpu_num)
        end = self.neuron_types*divceil(j*ommatidia_num, sublpu_num)
        return start, end

    def get_worker_interval(self, j, sublpu_num):
        '''
            Returns jth interval out of `sublpu_num`.
            Numbering starts from 1.
            Outputs are multiple of photoreceptor_types.
            End index of last interval might be greater
            but not less than the number of photoreceptors.
            Indexes match positions returned by
            `get_all_photoreceptors`
        '''
        ommatidia_num = self.num_elements
        start = self.photoreceptor_types*divceil((j-1)*ommatidia_num,
                                                 sublpu_num)
        end = self.photoreceptor_types*divceil(j*ommatidia_num, sublpu_num)
        return start, end

    # Node key representation
    def get_nodes(self, j, sublpu_num):
        # numbering starts from 1
        return list(range(*self.get_interval(j, sublpu_num)))

    def get_worker_nodes(self, j, sublpu_num):
        # numbering starts from 1
        return list(range(*self.get_worker_interval(j, sublpu_num)))

    # Selector representation
    def get_selectors(self, j, sublpu_num):
        # numbering starts from 1
        selectors = []
        for n in self.get_neurons(j, sublpu_num):
            n_id, n_name, _ = n
            selectors.append('/ret/{}/{}'.format(n_id, n_name))
        return selectors

    # Selector representation (all)
    def get_all_selectors(self):
        # Get interval as if there is 1 LPU
        # so get all nodes
        return self.get_selectors(1, 1)

    def get_master_selectors(self):
        selectors = []
        for i in self.get_worker_nodes(1, 1):
            col = i // 6
            ind = 1 + (i % 6)
            selectors.append('/master/{}/buf{}'.format(col, ind))
            selectors.append('/master/{}/R{}'.format(col, ind))

        return selectors

    def get_worker_selectors(self, j, workernum):
        selectors = []
        for i in self.get_worker_nodes(j, workernum):
            col = i // 6
            ind = 1 + (i % 6)
            selectors.append('/ret/{}/in{}'.format(col, ind))
            selectors.append('/ret/{}/R{}'.format(col, ind))

        return selectors

    # A convenient representation of all neuron information
    def get_neuron_objects(self):
        return self._ommatidia

    def get_neighborid(self, oid, neighbor_dr):
        ''' Get id of neighbor of `oid` ommatidium in a
            specific direction
        '''
        return self._ommatidia[oid].get_neighborid(neighbor_dr)

    def index(self, ommid, name):
        return self._ommatidia[ommid].neurons.keys().index(name)

    @property
    def num_neurons(self):
        return self.neuron_types*self.num_elements

    @property
    def num_photoreceptors(self):
        return self.photoreceptor_types*self.num_elements

    @property
    def num_elements(self):
        return self.hex_array.num_elements

    @property
    def neuron_types(self):
        return self._ommatidia[0].neuron_num

    @property
    def photoreceptor_types(self):
        return self._ommatidia[0].photoreceptor_num

    @property
    def radius(self):
        return self.hex_array.radius


class Neuron(object):
    def __init__(self, params):
        self.name = params.get('name')

        self.params = params.copy()
        self.num = None

        self.outgoing_synapses = []
        self.incoming_synapses = []

    def add_outgoing_synapse(self, synapse):
        self.outgoing_synapses.append(synapse)

    def add_incoming_synapse(self, synapse):
        self.incoming_synapses.append(synapse)

    def remove_outgoing_synapse(self, synapse):
        self.outgoing_synapses.remove(synapse)

    def remove_incoming_synapse(self, synapse):
        self.incoming_synapses.remove(synapse)

    def __repr__(self):
        return 'neuron {}: {}'.format(self.params['name'], self.params)

    def __str__(self):
        return 'neuron {}'.format(self.params['name'])


class OmmatidiumNeuron(Neuron):
    def __init__(self, ommatidium, direction, params, is_photoreceptor=True):
        '''
            ommatidium: ommatidium object
            direction: tuple of 2 coordinates (elevation, azimuth) or None
                       direction of photoreceptor optical axis
        '''
        self.parent = ommatidium
        self.direction = direction
        self._is_photoreceptor = is_photoreceptor

        super(OmmatidiumNeuron, self).__init__(params)

    @property
    def sphere_pos(self):
        return self.parent.sphere_pos
    @property
    def is_photoreceptor(self):
        return self._is_photoreceptor


class Synapse(object):
    def __init__(self, params):
        """ params: a dictionary of neuron parameters.
                    It can be passed as an attribute dictionary parameter
                    for a node in networkx library.
        """
        self._params = params.copy()

    def link(self, pre_neuron, post_neuron):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.pre_neuron.add_outgoing_synapse(self)
        self.post_neuron.add_incoming_synapse(self)

    def __repr__(self):
        return 'synapse from {} to {}: {}'.format(self.params['prename'],
                                                  self.params['postname'],
                                                  self.params)

    def __str__(self):
        return 'synapse from {} to {}'.format(self.params['prename'],
                                              self.params['postname'])

    @property
    def prenum(self):
        return self._prenum

    @prenum.setter
    def prenum(self, value):
        self._prenum = value

    @property
    def postnum(self):
        return self._postnum

    @postnum.setter
    def postnum(self, value):
        self._postnum = value

    @property
    def params(self):
        return self._params

    def process_before_export(self):
        # assumes all conductances are gpot to gpot
        self._params.update({'class': 3})
        self._params.update({'conductance': True})
        if 'cart' in self._params.keys():
            del self._params['cart']
        if 'scale' in self.params.keys():
            self._params['slope'] *= self._params['scale']
            self._params['saturation'] *= self._params['scale']
            del self._params['scale']


