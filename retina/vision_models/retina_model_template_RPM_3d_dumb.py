# RPM parameters
OMMATIDIA_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'class': 'RPM_3d_dumb',
        'init_V': -80., 'num_microvilli': 1
    }
    for i in range(6)
]

