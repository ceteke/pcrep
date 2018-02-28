import pickle, os, csv, joblib
import numpy as np
from latent_3d_points.src.in_out import PointCloudDataSet, create_dir
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.ae_templates import default_train_params, mlp_architecture_ala_iclr_18
from architectures import washington_xyz_rgb
import collections
from sklearn.model_selection import train_test_split
from utils import batchify
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

n_pc_points = 2048                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
experiment_name = 'shapenet_{}_{}'.format(n_pc_points, bneck_size)
train_dir = create_dir(os.path.join('data/', experiment_name))
train_params = default_train_params(single_class=False)
train_params['training_epochs'] = 5
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

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model('data/{}'.format(experiment_name), 350, True)

dataset = pickle.load(open('/home/ceteke/Desktop/demonstrations/dummy_xyz_500ms.pk', 'rb'))
files = pickle.load(open('/home/ceteke/Desktop/demonstrations/dummy_xyz_500ms_files.pk', 'rb'))

dataset_trans = ae.transform(dataset)

gmm = GaussianMixture(5)

pca = PCA(n_components=2)
dt = pca.fit_transform(dataset_trans)

gmm.fit(dt)
clusters = gmm.predict(dt)

colors = {0:'blue', 1:'red', 2:'green', 3:'black', 4:'cyan'}

for i, d in enumerate(dt):
    fp = files[i].split('/')
    fp = (fp[-2] + '/' + fp[-1]).split('_seg.pcd')[0]
    plt.scatter(d[0], d[1], color=colors[clusters[i]])
    # plt.annotate(fp, xy=(d[0], d[1]))
    print('{}: {}'.format(fp, clusters[i]))

plt.show()