# **** [vision_class_template.py]               ****
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

R_THRESHOLD = -70.0
L1_THRESHOLD = -50.5
L2_THRESHOLD = -50.5
L3_THRESHOLD = -50.5
L4_THRESHOLD = -50.5
L5_THRESHOLD = -50.5
C2_THRESHOLD = -50.0
C3_THRESHOLD = -50.0
AM_THRESHOLD = -70.0
T1_THRESHOLD = -50.0

ACH_RP = 0.0
HIST_RP = -80.0
GLU_RP = 0.0
GABA_RP = -80.0

AM_RP = GLU_RP

SLOPE = 0.02
SATURATION = 0.0008

OMMATIDIA_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'class': 'PhotoreceptorModel',
        'init_V': -81.9925, 'init_sa': 0.2184, 'init_si': 0.9653,
        'init_dra': 0.0117, 'init_dri': 0.9998, 'init_nov': 0.0017,
        'num_microvilli': 30000
    }
    for i in range(6)
]

CARTRIDGE_IN_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'class': 'Port',
        'port_type': 'gpot', 'port_io': 'in',
        'genetic.neurotransmitter': 'histamine'
    }
    for i in range(6)
]


CARTRIDGE_NEURON_LIST = [
    {
        'name': 'L1', 'class': 'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'glutamate'
    },
    {
        'name': 'L2', 'class': 'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'acetylcholine'
    },
    {
        'name': 'L3', 'class': 'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'glutamate'
    },
    {
        'name': 'L4', 'class': 'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'acetylcholine'
    },
    {
        'name': 'L5', 'class': 'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'glutamate'
    },
    {
        'name':'T1', 'class':'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'acetylcholine'
    },
    {
        'name':'C2', 'class':'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'GABA'
    },
    {
        'name':'C3', 'class':'MorrisLecar',
        'V1': -20.0, 'V2': 50.0, 'V3': -40.0, 'V4': 20.0,
        'V_L': -40.0, 'V_Ca': 120.0, 'V_K': -80.0,
        'g_L': 3.0, 'g_Ca': 4.0, 'g_K': 16.0,
        'phi': 0.004, 'initV': -46., 'initn': 0.35, 'offset': 0.0,
        'genetic.neurotransmitter': 'GABA'
    }
] + \
[
    {
        'name': 'a{}'.format(i+1)
    }
    for i in range(6)
]

AM_PARAMS = {
    'name': 'Am', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_K': -70.0,
        'g_L': 0.5, 'g_Ca': 1.1, 'g_K': 2.0,
        'phi': 0.04, 'initV': -50., 'initn': 0.5, 'offset': 0.02,
        'genetic.neurotransmitter': 'glutamate'
}


INTRA_CARTRIDGE_SYNAPSE_LIST = [
    {'prename':'R1', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':40, 'mode':0},
    {'prename':'R2', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':43, 'mode':0},
    {'prename':'R3', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':37, 'mode':0},
    {'prename':'R4', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':38, 'mode':0},
    {'prename':'R5', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':38, 'mode':0},
    {'prename':'R6', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':45, 'mode':0},

    {'prename':'R1', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':46, 'mode':0},
    {'prename':'R2', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':45, 'mode':0},
    {'prename':'R3', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':39, 'mode':0},
    {'prename':'R4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':41, 'mode':0},
    {'prename':'R5', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':39, 'mode':0},
    {'prename':'R6', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':47, 'mode':0},

    {'prename':'R1', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':11, 'mode':0},
    {'prename':'R2', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':10, 'mode':0},
    {'prename':'R3', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':4, 'mode':0},
    {'prename':'R4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':8, 'mode':0},
    {'prename':'R5', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':6, 'mode':0},
    {'prename':'R6', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.01,
    'scale':12, 'mode':0},
    
    {'prename':'L2', 'postname':'R1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':2,
    'threshold':L2_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.0016,
    'scale':1, 'mode':0},
    {'prename':'L2', 'postname':'R2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':2,
    'threshold':L2_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.0016,
    'scale':1, 'mode':0},
    {'prename':'L2', 'postname':'L1', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.001, 'power':1.0, 'saturation':0.02,
    'scale':3, 'mode':0},
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':4, 'mode':0},
    {'prename':'L2', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},

    {'prename':'L4', 'postname':'R5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':2,
    'threshold':L4_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.0016,
    'scale':1, 'mode':0},
    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':2,
    'threshold':L4_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.0016,
    'scale':3, 'mode':0},
    {'prename':'L4', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':2,
    'threshold':L4_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.0016,
    'scale':1, 'mode':0},

    {'prename':'R1', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':19, 'mode':0},
    {'prename':'R2', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':16, 'mode':0},
    {'prename':'R2', 'postname':'a2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':22, 'mode':0},
    {'prename':'R3', 'postname':'a2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':18, 'mode':0},
    {'prename':'R3', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':20, 'mode':0},
    {'prename':'R4', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':16, 'mode':0},
    {'prename':'R4', 'postname':'a4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':17, 'mode':0},
    {'prename':'R5', 'postname':'a4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':26, 'mode':0},
    {'prename':'R5', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':10, 'mode':0},
    {'prename':'R6', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':14, 'mode':0},
    {'prename':'R6', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':22, 'mode':0},
    {'prename':'R1', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.005,
    'scale':17, 'mode':0},

    {'prename':'a1', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':1, 'mode':0},
    {'prename':'a2', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':1, 'mode':0},
    {'prename':'a3', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':3, 'mode':0},
    {'prename':'a5', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':5, 'mode':0},
    {'prename':'a6', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.04,
    'scale':1, 'mode':0},

    {'prename':'a5', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':3, 'mode':0},
    {'prename':'a4', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'L5', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':4, 'mode':0},

    {'prename':'a1', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':8, 'mode':0},
    {'prename':'a2', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':6, 'mode':0},
    {'prename':'a3', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':7, 'mode':0},
    {'prename':'a4', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':12, 'mode':0},
    {'prename':'a5', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':3, 'mode':0},
    {'prename':'a6', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':13, 'mode':0},
    
    {'prename':'a2', 'postname':'R2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a3', 'postname':'R4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'R4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'R5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a5', 'postname':'R5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    
    {'prename':'C2', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':1, 'mode':0},
    {'prename':'C2', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':2, 'mode':0},
    {'prename':'C2', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':1, 'mode':0},
    {'prename':'C2', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':1, 'mode':0},
    {'prename':'C3', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':1, 'mode':0},
    {'prename':'C3', 'postname':'a2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':3, 'mode':0},
    {'prename':'C3', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':2, 'mode':0},
    {'prename':'C3', 'postname':'a4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':2, 'mode':0},
    {'prename':'C3', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':2, 'mode':0},
    {'prename':'C3', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':3, 'mode':0},
    
    {'prename':'L1', 'postname':'C2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':GLU_RP, 'delay':1,
    'threshold':L1_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':3, 'mode':0},
    {'prename':'L1', 'postname':'C3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':GLU_RP, 'delay':1,
    'threshold':L1_THRESHOLD, 'slope':0.0002, 'power':1.0, 'saturation':0.006,
    'scale':5, 'mode':0},
    
    
    
]

CARTRIDGE_CR_II_SYNAPSE_LIST = [
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':6, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':4, 'mode':0},
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':5, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':2, 'mode':0},

    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',#
    'cart':3, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.1,
    'scale':4, 'mode':0},
    {'prename':'L4', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':3, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.001, 'power':1.0, 'saturation':0.5,
    'scale':1, 'mode':0},

    {'prename':'L4', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':4, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.05,
    'scale':2, 'mode':0},

    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.2,
    'scale':2, 'mode':0},
    {'prename':'L4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.05,
    'scale':1, 'mode':0},
    {'prename':'L4', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.05,
    'scale':1, 'mode':0},
    {'prename':'L4', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.1,
    'scale':1, 'mode':0},
]

