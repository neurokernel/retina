# **** [retina_model_template.py]               ****
# **** Do not modify this file                  ****
# **** Copy into a new file and modify that one ****

# Photoreceptor parameters
OMMATIDIA_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'model': 'Photoreceptor',
        'public': True, 'extern': True, 'spiking': False,
        'init_V': -81.99, 'init_sa': 0.2184, 'init_si': 0.9653,
        'init_dra': 0.0117, 'init_dri': 0.9998, 'init_nov': 0.0017,
        'num_microvilli': 30000
    }
    for i in range(6)
]

