from abc import ABCMeta, abstractmethod


class OpticAxisRule(object):
    # return values of neighbor_for_photor and
    # neighbor_that_provide_photor should be the
    # following:
    # [x] is the neighbor at position x
    # [x, y] is the neighbor at position y
    # of neighbor at position x
    # this is commutative so order does not matter
    # positions are relative to o.
    # The actual directions are given by `_get_unit_axis`
    # of `HexagonArray` class
    #       1
    #   6       2
    #       o
    #   5       3
    #       4
    # 0 is the current column, arrangement is based
    # on right eye as seen from inside or left eye as
    # seen from outside.
    # 5 and 6 is the anterior side and 2, 3 the posterior
    # dorsal/ventral depend on whether it is the left or
    # the right eye.
    # For the left eye, 1 is dorsal.
    # For the right eye, 4 is dorsal.
    # See also RFC #3 for the coordinate system for the left eye.
    # The right eye is constructed by a rotation in opposit direction
    # of the rotation of left eye.
    __metaclass__ = ABCMeta

    inds = {'R{}'.format(i+1): i+1 for i in range(6)}

    @classmethod
    def name_to_ind(cls, name):
        try:
            return cls.inds[name]
        except KeyError:
            print('"{}" is not a valid neuron name'.format(name))
            raise

    @classmethod
    def is_photor(cls, name):
        return name in cls.inds.keys()

    @abstractmethod
    def neighbor_for_photor(self, photor_ind):
        return

    @abstractmethod
    def neighbor_that_provide_photor(self, photor_ind):
        return

# same as Right Bottom
class OpticAxisNeuralSuperpositionLeftTop(OpticAxisRule):
    def __init__(self):
        self.name = 'neural superposition'

    def neighbor_for_photor(self, photor_ind):
        if photor_ind == 1:
            neighbor = [2]
        elif photor_ind == 2:
            neighbor = [3]
        elif photor_ind == 3:
            neighbor = [3, 4]
        elif photor_ind == 4:
            neighbor = [4]
        elif photor_ind == 5:
            neighbor = [5]
        elif photor_ind == 6:
            neighbor = [6]
        else:
            raise ValueError('Unexpected neighbor index {}. Expected 1-6.'
                             .format(photor_ind))
        return neighbor

    def neighbor_that_provide_photor(self, photor_ind):
        # like neighbor_for_photor
        if photor_ind == 1:
            neighbor = [5]
        elif photor_ind == 2:
            neighbor = [6]
        elif photor_ind == 3:
            neighbor = [1, 6]
        elif photor_ind == 4:
            neighbor = [1]
        elif photor_ind == 5:
            neighbor = [2]
        elif photor_ind == 6:
            neighbor = [3]
        return neighbor

# same as Right Top
class OpticAxisNeuralSuperpositionLeftBottom(OpticAxisRule):
    def __init__(self):
        self.name = 'neural superposition'

    def neighbor_for_photor(self, photor_ind):
        if photor_ind == 1:
            neighbor = [3]
        elif photor_ind == 2:
            neighbor = [2]
        elif photor_ind == 3:
            neighbor = [1, 2]
        elif photor_ind == 4:
            neighbor = [1]
        elif photor_ind == 5:
            neighbor = [6]
        elif photor_ind == 6:
            neighbor = [5]
        else:
            raise ValueError('Unexpected neighbor index {}. Expected 1-6.'
                             .format(photor_ind))
        return neighbor

    def neighbor_that_provide_photor(self, photor_ind):
        # like neighbor_for_photor
        if photor_ind == 1:
            neighbor = [6]
        elif photor_ind == 2:
            neighbor = [5]
        elif photor_ind == 3:
            neighbor = [4, 5]
        elif photor_ind == 4:
            neighbor = [4]
        elif photor_ind == 5:
            neighbor = [3]
        elif photor_ind == 6:
            neighbor = [2]
        return neighbor

# same as Left Bottom
class OpticAxisNeuralSuperpositionRightTop(OpticAxisRule):
    def __init__(self):
        self.name = 'neural superposition'

    def neighbor_for_photor(self, photor_ind):
        if photor_ind == 1:
            neighbor = [3]
        elif photor_ind == 2:
            neighbor = [2]
        elif photor_ind == 3:
            neighbor = [1, 2]
        elif photor_ind == 4:
            neighbor = [1]
        elif photor_ind == 5:
            neighbor = [6]
        elif photor_ind == 6:
            neighbor = [5]
        else:
            raise ValueError('Unexpected neighbor index {}. Expected 1-6.'
                             .format(photor_ind))
        return neighbor

    def neighbor_that_provide_photor(self, photor_ind):
        # like neighbor_for_photor
        if photor_ind == 1:
            neighbor = [6]
        elif photor_ind == 2:
            neighbor = [5]
        elif photor_ind == 3:
            neighbor = [4, 5]
        elif photor_ind == 4:
            neighbor = [4]
        elif photor_ind == 5:
            neighbor = [3]
        elif photor_ind == 6:
            neighbor = [2]
        return neighbor

# same as Left Top
class OpticAxisNeuralSuperpositionRightBottom(OpticAxisRule):
    def __init__(self):
        self.name = 'neural superposition'

    def neighbor_for_photor(self, photor_ind):
        if photor_ind == 1:
            neighbor = [2]
        elif photor_ind == 2:
            neighbor = [3]
        elif photor_ind == 3:
            neighbor = [3, 4]
        elif photor_ind == 4:
            neighbor = [4]
        elif photor_ind == 5:
            neighbor = [5]
        elif photor_ind == 6:
            neighbor = [6]
        else:
            raise ValueError('Unexpected neighbor index {}. Expected 1-6.'
                             .format(photor_ind))
        return neighbor

    def neighbor_that_provide_photor(self, photor_ind):
        if photor_ind == 1:
            neighbor = [5]
        elif photor_ind == 2:
            neighbor = [6]
        elif photor_ind == 3:
            neighbor = [1, 6]
        elif photor_ind == 4:
            neighbor = [1]
        elif photor_ind == 5:
            neighbor = [2]
        elif photor_ind == 6:
            neighbor = [3]
        return neighbor


class OpticAxisPlain(OpticAxisRule):
    def __init__(self):
        self.name = 'plain'

    def neighbor_for_photor(self, photor_ind):
        return [0]

    def neighbor_that_provide_photor(self, photor_ind):
        return [0]


class RuleHexArrayMap(object):
    '''
        A class that assigns columns based on composition rule
        in a consistent way.
    '''
    def __init__(self, rule, hexarray):
        # keys are tuples (column_id, photoreceptorname)
        neighbors_for_photor = {}
        neighbors_that_provide_photor = {}

        for el in hexarray.elements:
            for neuron, ind in OpticAxisRule.inds.items():
                neighbordr = rule.neighbor_for_photor(ind)
                neighborid = hexarray.get_neighborid(el.gid, neighbordr)

                # While given selector is connected to a port
                # search for a selector of a cartridge
                # in the opposite direction (than the original rule).
                # Stop if selector is the same as the previous.
                # Here we assume the 2 methods of opticaxis
                # return cartridge ids in opposite directions and if
                # there is no cartridge in one of those directions
                # method returns the current cartridge id
                # instead of the neighbor's.
                # In case the rule connects ommatidium and cartridges
                # with same ids there should be no conflict in the first
                # place (the following check is always False)
                # Another option is to ignore those connections
                # but unconnected ports cause problems elsewhere
                while (neighborid, neuron) in \
                        neighbors_that_provide_photor.keys():
                    neighbordr = rule.neighbor_that_provide_photor(ind)
                    neighborid_new = hexarray.get_neighborid(neighborid, neighbordr)
                    if neighborid_new == neighborid:
                        break
                    else:
                        neighborid = neighborid_new

                neighbors_for_photor[(el.gid, neuron)] = neighborid
                neighbors_that_provide_photor[(neighborid, neuron)] = el.gid

        self.neighbors_for_photor = neighbors_for_photor
        self.neighbors_that_provide_photor = neighbors_that_provide_photor

    def neighbor_for_photor(self, column_id, photor):
        return self.neighbors_for_photor[(column_id, photor)]

    def neighbor_that_provide_photor(self, column_id, photor):
        return self.neighbors_that_provide_photor[(column_id, photor)]

# don't use this directly
# implementation might change
_class_dict = {
    'Plain': OpticAxisPlain,
    'SuperpositionLT': OpticAxisNeuralSuperpositionLeftTop,
    'SuperpositionLB': OpticAxisNeuralSuperpositionLeftBottom,
    'SuperpositionRT': OpticAxisNeuralSuperpositionRightTop,
    'SuperpositionRB': OpticAxisNeuralSuperpositionRightBottom
}


def opticaxisFactory(rule):
    try:
        return _class_dict[rule]
    except KeyError:
        raise ValueError('Value {} not in axis rules {}'
                         ' dictionary'.format(rule, _class_dict.keys()))


def main():
    axis = opticaxisFactory('Superposition')()
    neuronname = 'R1'
    ind = axis.name_to_ind(neuronname)
    print(axis.neighbor_for_photor(ind))

if __name__ == '__main__':
    main()
