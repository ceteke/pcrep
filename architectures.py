'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from latent_3d_points.src.encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only


def washington_xyz_rgb(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args
