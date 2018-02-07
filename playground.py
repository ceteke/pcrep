import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph

top_out_dir = 'data/'                        # Use to write Neural-Net check-points etc.

experiment_name = 'shapenet_2048_32'
n_pc_points = 2048                              # Number of points per model.
bneck_size = 32                                # Bottleneck-AE size
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
dataset_dir = '/home/ceteke/Documents/datasets/Shapenet/pc_2048/shape_net_core_uniform_samples_2048'
print("Loading dataset..")
all_pc_data = load_all_point_clouds_under_folder(dataset_dir, n_threads=8, file_ending='.ply', verbose=True)

train_dir = create_dir(osp.join(top_out_dir, experiment_name))
train_params = default_train_params(single_class=False)
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

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
conf.held_out_step = 5              # How often to evaluate/print out loss on held_out data (if any).
conf.save(osp.join(train_dir, 'configuration'))

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
print("Training...")
buf_size = 1 # flush each line
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()
