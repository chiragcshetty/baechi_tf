# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from image_classifier.networks import cifarnet
from image_classifier.networks import inception_v3
from image_classifier.networks import nasnet
from image_classifier.networks import pnasnet

slim = tf.contrib.slim

networks_map = {
    'cifarnet': cifarnet.cifarnet,
    'inception_v3': inception_v3.inception_v3,
    'nasnet_mobile':nasnet.build_nasnet_mobile,
    'nasnet_large':nasnet.build_nasnet_large,
    'nasnet_cifar':nasnet.build_nasnet_cifar,
    'pnasnet_mobile':pnasnet.build_pnasnet_mobile,
    'pnasnet_large':pnasnet.build_pnasnet_large,
}

arg_scopes_map = {
    'cifarnet': cifarnet.cifarnet_arg_scope,
    'inception_v3': inception_v3.inception_v3_arg_scope,
    'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
    'nasnet_large': nasnet.nasnet_large_arg_scope,
    'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
    'pnasnet_mobile': pnasnet.pnasnet_mobile_arg_scope,
    'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes=num_classes, is_training=is_training,
                  **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
