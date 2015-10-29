# **** [retina_model_template.py]               ****
# **** Do not modify this file                  ****
# **** Copy into a new file and modify that one ****

# Photoreceptor parameters
OMMATIDIA_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'model': 'Photoreceptor',
        'public': True, 'extern': True, 'spiking': False,
        'init_V': -68.7015, 'init_sa': 0.3068, 'init_si': 0.9101,
        'init_dra': 0.0242, 'init_dri': 0.9988, 'num_microvilli': 30000
    }
    for i in range(6)
]

