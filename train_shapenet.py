import pickle, os, csv, joblib
import numpy as np
from latent_3d_points.src.in_out import PointCloudDataSet, create_dir
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.ae_templates import default_train_params
from architectures import washington_xyz_rgb
from utils import batchify
from datasets import ShapeNet

shapenet_dataset = ShapeNet('/home/ceteke/Documents/datasets/Shapenet/pc_1024_py2', 10)
X = shapenet_dataset.process()
print(X.shape)

n_pc_points = 1024                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
experiment_name = 'shapenet_1024_ae_128'
train_dir = create_dir(os.path.join('data/', experiment_name))
train_params = default_train_params(single_class=False)
train_params['training_epochs'] = 100
encoder, decoder, enc_args, dec_args = washington_xyz_rgb(n_pc_points, bneck_size)

pcd_dataset = PointCloudDataSet(X, copy=False)

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )

conf.experiment_name = experiment_name
conf.save(os.path.join(train_dir, 'configuration'))

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

buf_size = 1 # flush each line
fout = open(os.path.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(pcd_dataset, conf, log_file=fout)
fout.close()