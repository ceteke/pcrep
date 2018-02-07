# -*- coding: utf-8 -*-

from datasets import Tork18
import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import write_pcd
import numpy as np

top_out_dir = 'data/'                        # Use to write Neural-Net check-points etc.


#conf.save(osp.join(train_dir, 'configuration'))

sizes = [32, 64, 128]
train_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8]
num_points = []
accs = {32: [], 64: [], 128: []}
folds = 10

for size in sizes:
  for t in train_sizes:

    experiment_name = 'shapenet_2048_{}'.format(size)
    n_pc_points = 2048  # Number of points per model.
    bneck_size = size  # Bottleneck-AE size
    ae_loss = 'emd'  # Loss to optimize: 'emd' or 'chamfer'

    train_dir = create_dir(osp.join(top_out_dir, experiment_name))
    train_params = default_train_params(single_class=False)
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

    conf = Conf(n_input=[n_pc_points, 3],
                loss=ae_loss,
                training_epochs=train_params['training_epochs'],
                batch_size=train_params['batch_size'],
                denoising=train_params['denoising'],
                learning_rate=train_params['learning_rate'],
                train_dir=train_dir,
                loss_display_step=train_params['loss_display_step'],
                saver_step=train_params['saver_step'],
                z_rotate=train_params['z_rotate'],
                encoder=encoder,
                decoder=decoder,
                encoder_args=enc_args,
                decoder_args=dec_args
                )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5  # How often to evaluate/print out loss on held_out data (if any).

    reset_tf_graph()
    ae = PointNetAutoEncoder(experiment_name, conf)

    t18 = Tork18('/home/ceteke/Documents/datasets/tork18', 'pour')
    X, y = t18.get_dataset()

    if len(num_points) != len(train_sizes):
      num_points.append(int(len(X) * t))

    ae.restore_model(osp.join(top_out_dir, experiment_name), 100, True)

    scores = 0.0
    for _ in range(folds):
      X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t, shuffle=True)

      print(len(X_train), len(X_test))

      svm = LinearSVC()

      X_train_trans = ae.transform(X_train)
      X_test_trans = ae.transform(X_test)

      #markers = {0: 'o', 1: '+'}
      #colors = {0: 'red', 1: 'blue'}
      #labels = {0: 'Once', 1: 'Sonra'}

      svm.fit(X_train_trans, y_train)
      scores += svm.score(X_test_trans, y_test)

    a = accs[size]
    a.append(scores/folds)
    accs[size] = a

for k, v in accs.items():
  print(k, v, num_points)
  plt.plot(num_points, v, label=str(k))
plt.legend()

#pca = PCA(2)
#X_trans = pca.fit_transform(ae.transform(X))

#X_recons, _ = ae.reconstruct(X)
#write_pcd('orig.pcd', X[80])
#write_pcd('recons.pcd', X_recons[80])

#plt.scatter(X_trans[np.where(y==0)[0]][:,0], X_trans[np.where(y==0)[0]][:,1], marker='o', label=u'Başarısız')
#plt.scatter(X_trans[np.where(y==1)[0]][:,0], X_trans[np.where(y==1)[0]][:,1], marker='+', label=u'Başarılı')

#plt.legend(loc="upper left")
plt.savefig('pour_trains.png', bbox_inches='tight')