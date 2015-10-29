#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='neurokernel-retina',
      version='1.0',
      packages=['retina', 'retina.vision_models',
                'retina.neurons', 'retina.synapses', 'retina.geometry'],
      install_requires=[
        'configobj >= 5.0.0',
        'neurokernel >= 0.1'
      ]
     )
