import screen.screen as scr
import screen.map.mapimpldr as mapdr
import vrf.vrf as vrf
import vrf.vrf_no_gpu as vrfn

CYLINDER = 'Cylinder'
SPHERE = 'Sphere'


_scr_class_dict = {
    CYLINDER: scr.CylinderScreen,
    SPHERE: scr.SphereScreen
}


def get_screen_cls(screen):
    try:
        return _scr_class_dict[screen]
    except KeyError:
        raise ValueError('Value {} not in screen types: {}'
                         .format(screen, _scr_class_dict.keys()))


_vrf_class_dict = {
    CYLINDER: vrf.Cylinder_Gaussian_RF,
    SPHERE: vrf.Sphere_Gaussian_RF
}


def get_vrf_cls(vrf_type):
    try:
        return _vrf_class_dict[vrf_type]
    except KeyError:
        raise ValueError('Value {} not in vrf types: {}'
                         .format(vrf_type, _vrf_class_dict.keys()))

_vrfn_class_dict = {
    CYLINDER: vrfn.Cylinder_Gaussian_RF,
    SPHERE: vrfn.Sphere_Gaussian_RF
}


def get_vrf_no_gpu_cls(vrfn_type):
    try:
        return _vrfn_class_dict[vrfn_type]
    except KeyError:
        raise ValueError('Value {} not in vrf_no_gpu types: {}'
                         .format(vrfn_type, _vrfn_class_dict.keys()))




_mapdr_class_dict = {
    CYLINDER: mapdr.SphereToCylinderMap,
    SPHERE: mapdr.SphereToSphereMap
}


def get_mapdr_cls(map_type):
    try:
        return _mapdr_class_dict[map_type]
    except KeyError:
        raise ValueError('Value {} not in map types: {}'
                         .format(map_type, _mapdr_class_dict.keys()))
