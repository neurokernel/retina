# **** [vision_model_template.py]               ****
# **** Do not modify this file                  ****
# **** Copy into a new file and modify that one ****

# graded potential to graded potential synapses documented in
# Neurokernel RFC 2 http://neurokernel.github.io/rfc/nk-rfc2.pdf
# RFC   | Here
# ------------------
# k     | slope
# V_th  | threshold
# n     | power
# g_sat | saturation

R_THRESHOLD = -50.0
L2_THRESHOLD = -50.0
AM_THRESHOLD = -50.0

ACH_RP = -10.0
HIST_RP = -80.0
GLU_RP = -10.0
GABA_RP = -80.0

AM_RP = GLU_RP

SLOPE = 0.0001  # 0.02
SATURATION = 0.001

OMMATIDIA_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'model': 'Photoreceptor',
        'public': True, 'extern': True, 'spiking': False,
        'init_V': -68.7015, 'init_sa': 0.3068, 'init_si': 0.9101,
        'init_dra': 0.0242, 'init_dri': 0.9988, 'num_microvilli': 30000
    }
    for i in range(6)
]

CARTRIDGE_IN_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'model': 'port_in_gpot',
        'public': False, 'extern': True
    }
    for i in range(6)
]

CARTRIDGE_NEURON_LIST = [
   {
        'name': 'L2', 'model': 'MorrisLecar',
        'extern': False, 'public': True, 'spiking': False,
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_l': -40.0, 'V_ca': 120.0, 'V_k': -80.0,
        'G_l': 3.0, 'G_ca': 4.0, 'G_k': 16.0,
        'phi': 0.001, 'initV': -46.424, 'initn': 0.3447, 'offset': 0.0
    }
] + \
[
    {
        'name': 'a{}'.format(i+1)
    }
    for i in range(6)
]

AM_PARAMS = {
    'name': 'Am', 'model': 'MorrisLecar',
    'extern': False, 'public': True, 'spiking': False,
    'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
    'V_l': -40.0, 'V_ca': 120.0, 'V_k': -80.0,
    'G_l': 3.0, 'G_ca': 4.0, 'G_k': 16.0,
    'phi': 0.001, 'initV': -48.57, 'initn': 0.3525, 'offset': 0.0
}


INTRA_CARTRIDGE_SYNAPSE_LIST = [
    {'prename':'R1', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':46, 'mode':0},
    {'prename':'R2', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':45, 'mode':0},
    {'prename':'R3', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':39, 'mode':0},
    {'prename':'R4', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':41, 'mode':0},
    {'prename':'R5', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':39, 'mode':0},
    {'prename':'R6', 'postname':'L2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':47, 'mode':0},

    {'prename':'L2', 'postname':'R1', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':ACH_RP, 'delay':2,
    'threshold':L2_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'L2', 'postname':'R2', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':ACH_RP, 'delay':2,
    'threshold':L2_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},

    {'prename':'R1', 'postname':'a1', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':19, 'mode':0},
    {'prename':'R2', 'postname':'a1', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':16, 'mode':0},
    {'prename':'R2', 'postname':'a2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':22, 'mode':0},
    {'prename':'R3', 'postname':'a2', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':18, 'mode':0},
    {'prename':'R3', 'postname':'a3', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':20, 'mode':0},
    {'prename':'R4', 'postname':'a3', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':16, 'mode':0},
    {'prename':'R4', 'postname':'a4', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':17, 'mode':0},
    {'prename':'R5', 'postname':'a4', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':26, 'mode':0},
    {'prename':'R5', 'postname':'a5', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':10, 'mode':0},
    {'prename':'R6', 'postname':'a5', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':14, 'mode':0},
    {'prename':'R6', 'postname':'a6', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':22, 'mode':0},
    {'prename':'R1', 'postname':'a6', 'model':'PowerGpotGpotSig',
    'cart':None, 'V_rev':HIST_RP, 'delay':2,
    'threshold':R_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':17, 'mode':0},

    {'prename':'a6', 'postname':'R1', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'a2', 'postname':'R2', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'a3', 'postname':'R4', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'R4', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'R5', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
    {'prename':'a5', 'postname':'R5', 'model':'PowerGpotGpotSig',#
    'cart':None, 'V_rev':AM_RP, 'delay':2,
    'threshold':AM_THRESHOLD, 'slope':0.5*SLOPE, 'power':1.0, 'saturation':SATURATION,
    'scale':1, 'mode':0},
]

CARTRIDGE_CR_II_SYNAPSE_LIST = [
]

