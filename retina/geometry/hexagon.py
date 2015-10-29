from __future__ import unicode_literals
from __future__ import division

import numpy as np
PI = np.pi


class ArrayElement(object):
    def __init__(self, gid, dima, dimb):
        self.gid = gid
        self.dima = dima
        self.dimb = dimb
        self.neighbors = [self]

    @property
    def is_dummy(self):
        return self.gid < 0

    def get_neighborid(self, neighbor_dr):
        '''
            Returns global id of neighbor element
            or of current element if neighbor
            does not exist

            neighbor_dr: a list of directions(1-6) on
            hexagonal grid that one has to follow
            to reach neighbor. 0 corresponds to current
            ommatidium
            (see `_get_unit_axis` of `HexagonArray` class)
        '''
        neighbor_el = self

        for dr in neighbor_dr:
            neighbor_el = neighbor_el.neighbors[dr]

        if neighbor_el.is_dummy:
            return self.gid
        else:
            return neighbor_el.gid

class HexagonArray(object):

    def __init__(self, **kwargs):
        # total number of rings used
        self._num_rings = int(kwargs.get('num_rings', 14))
        # radius of the ring
        self._radius = float(kwargs.get('radius', 1))
        # transformation
        self._transform = kwargs.get('transform', None)

        # add 2 rings as buffer rings
        # add extra rings to include all cartridges in a certain radius
        # in practise with the addition of the 2 rings the extra rings
        # are useful only for large numbers of rings but additional rings
        # are filtered out anyway
        self._all_rings = self._num_rings + 2
        self._extra_rings = int(np.ceil(self._all_rings * 2. / np.sqrt(3)))

        # generate identifiers of elements of hexagon array
        # that can determine their position
        self._generate_ids()

        # generate positions of elements of hexagon array.
        self._generate_pos()

        # Keep elements of hexagon array in a disc of certain radius
        self._filter_elements()

        # generate element objects
        self._set_elements()

        # store all neighbors
        self._generate_neighbors()

    # total number of elements in the hexagon array.
    @property
    def num_elements(self):
        return self._rids.size

    # total number of elements in the hexagon array.
    @property
    def radius(self):
        return self._radius

    @property
    def hex_loc(self):
        return self._loc

    def _generate_ids(self):
        # ring ids
        self._rids = np.concatenate(
            [np.zeros(i*6 if i > 0 else 1, dtype=np.int32)+i
             for i in range(self._extra_rings+1)])

        # local ids in each ring
        self._lids = np.concatenate(
            [np.arange(0, i*6 if i > 0 else 1, dtype=np.int32)
             for i in range(self._extra_rings+1)])

        np.seterr(divide='ignore')
        # section ids, each ring is divided into 6 sections
        # each corresponding to an edge of the ring
        self._sids = np.divide(self._lids, self._rids)
        # section local ids, the number from the first element of the section.
        self._mids = np.mod(self._lids, self._rids)
        np.seterr(divide='warn')

    def _set_elements(self):
        if self._transform is not None:
            dima, dimb = self._transform(self.hex_loc[:, 0],
                                         self.hex_loc[:, 1])
        else:
            dima, dimb = self.hex_loc[:, 0], self.hex_loc[:, 1]

        self.elements = [ArrayElement(i, dima[i], dimb[i])
                         for i in range(self.num_elements)]
        self._make_dummy()

    def _make_dummy(self):
        # make dummy element and point neurons without a neighbor to it
        self.dummy = ArrayElement(-1, 0, 0)
        for i in range(6):
            self.dummy.neighbors.append(self.dummy)

    def _generate_pos(self):
        """ calculate position of all elements """

        # get the first 3 unit directions of hexagon grid (2 coordinates)
        dd = np.vstack([self._get_unit_axis(i+1) for i in range(3)])
        ref_dirs = np.asarray([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]], np.int32)
        ref_locs = np.asarray([[0, 0, 1],
                               [-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1],
                               [1, 0, 0],
                               [0, 1, 0]], np.int32)

        # rid -> ring id, sid -> section id(0-6), mid -> section local id
        # elementwise multiplication, 1st term vector from origin to
        # a ring section reference point, 2nd term vector from section
        # reference point to local ring point
        # The example shows the reference ommatidium at center(c) a point
        # at ring #2 at down reference direction (1)
        # and a point after adding a local direction in lower section(2)
        #           x
        #       x       x
        #   x       x       x
        #       x       x
        #   x       c       x
        #       x       x
        #   x       x       x
        #       2       x
        #           1
        v = ref_dirs[self._sids, :] * np.tile(
            self._rids.reshape(-1, 1), [1, 3]) + \
            ref_locs[self._sids, :] * np.tile(
            self._mids.reshape(-1, 1), [1, 3])

        # translation of each datapoint from 3D hex coordinates
        # (1 for each direction) to 2D cartesian coordinates
        self._loc = np.dot(v, dd)

    def _filter_elements(self):
        # Returns elements in a circle
        ind = (
            np.linalg.norm(self._loc, axis=1) <=
            (self._num_rings*self._get_column_d())).nonzero()[0]
        self._rids = self._rids[ind]
        self._lids = self._lids[ind]
        self._sids = self._sids[ind]
        self._mids = self._mids[ind]
        self._loc = self._loc[ind, :]

    def _get_column_d(self):
        """ get length between two neighboring elements """
        return self._radius/self._all_rings

    def _get_unit_axis(self, axis):
        """
        get unit direction of each axis

        Parameters
        ----------
        axis : int
            1: up
            2: upper-right
            3: lower-right
            4: down
            5: lower-left
            6: upper-left
        """
        r = self._get_column_d()
        if axis == 1:
            d = np.asarray([0, r], dtype=np.double)
        elif axis == 2:
            d = np.asarray([r*0.5*np.sqrt(3), r*0.5], dtype=np.double)
        elif axis == 3:
            d = np.asarray([r*0.5*np.sqrt(3), -r*0.5], dtype=np.double)
        elif axis == 4:
            d = np.asarray([0, -r], dtype=np.double)
        elif axis == 5:
            d = np.asarray([-r*0.5*np.sqrt(3), -r*0.5], dtype=np.double)
        elif axis == 6:
            d = np.asarray([-r*0.5*np.sqrt(3), r*0.5], dtype=np.double)
        else:
            raise ValueError('Axis value {} is not  in [1,6]'.format(axis))
        return d

    def _generate_neighbors(self):
        """ update the list of neighbors for all elements """
        allind = np.arange(self.num_elements, dtype=np.int32)
        for j in range(6):
            # return indices of columns and neighbor columns
            # in a specific direction
            col, col_n = self._find_neighbors(j+1)
            for icol, icol_n in zip(col, col_n):
                self.elements[icol].neighbors.append(self.elements[icol_n])
            # iterate over indices that are *not* in col (invert:True)
            for i in allind[np.in1d(allind, col, assume_unique=True,
                                    invert=True)]:
                self.elements[i].neighbors.append(self.dummy)

        # should not be true
        if not np.all(np.array([len(el.neighbors)
                                for el in self.elements]) == 7):
            raise ValueError("Error in updating neighbor lists")

    def _find_neighbors(self, pos):
        """
            Find the neighbor at relative position pos
            Returns a tuple with 2 arrays of equal length
            First is the indexes of columns
            and second the indexes of neighbors
        """
        d = self._get_unit_axis(pos)

        neighbor_loc = d + self.hex_loc

        # Compares all pairwise distances of locations
        # and shifted locations to the neighbor direction
        # 0.1 is the tolerance
        return (np.linalg.norm(neighbor_loc[:, None, :]
                               - self.hex_loc[None, :, :],
                               axis=2)
                < (self._get_column_d()*0.1)).nonzero()

    def get_config(self):
        config = {'num_rings': self._num_rings, 'radius': self._radius,
                  'all_rings': self._all_rings}
        return config

    def get_neighborid(self, elid, neighbor_dr):
        ''' Get id of neighbor of `elid` element in a
            specific direction
        '''
        return self.elements[elid].get_neighborid(neighbor_dr)


def main():
    a = HexagonArray(num_rings=14)
    print(a.num_elements)


if __name__ == "__main__":
    main()
