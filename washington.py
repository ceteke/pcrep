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

def load_points(path, label_id, label):
  x = pickle.load(open(os.path.join(path, label), 'rb'))
  if len(x) == 0:
    print('{} has no examples. Maybe all of the data has less than 1024 points?'.format(label))
    return None
  return x

def read_dataset(path):
  with open(os.path.join(path, 'labels.txt'), 'r') as f:
    reader = csv.reader(f)
    with joblib.Parallel(n_jobs=8, verbose=1) as parallel:
      dataset = parallel(joblib.delayed(load_points)(path, row[0], row[1]) for row in reader)

  dataset = list(filter(lambda x: x is not None, dataset))
  y = [np.ones(dataset[i].shape[0]) * i for i in range(len(dataset))]

  X = np.concatenate(dataset)
  y = np.concatenate(y)
  return X, y

print("Loading dataset...")
X, y = read_dataset('/home/ceteke/Documents/datasets/washington_xyzrgb')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=521)
c1 = collections.Counter(y_train.tolist())
print([(i, float(c1[i]) / len(y_train) * 100.0) for i in c1])
c2 = collections.Counter(y_test.tolist())
print([(i, float(c2[i]) / len(y_test) * 100.0) for i in c2])
X_train = X_train[:,:,:3]
X_test = X_test[:,:,:3]
print("Train shape", X_train.shape)
print("Test shape", X_test.shape)

n_pc_points = 1024                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
experiment_name = 'shapenet_1024_ae_128'
train_dir = create_dir(os.path.join('data/', experiment_name))
train_params = default_train_params(single_class=False)
train_params['training_epochs'] = 5
encoder, decoder, enc_args, dec_args = washington_xyz_rgb(n_pc_points, bneck_size)

pcd_dataset = PointCloudDataSet(X_train, labels=y_train, copy=False)

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

#buf_size = 1 # flush each line
#fout = open(os.path.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
#train_stats = ae.train(pcd_dataset, conf, log_file=fout)
#fout.close()

ae.restore_model('data/shapenet_1024_ae_128', 90, True)

print("Transforming Training data")
X_train_trans = []
for x_b in batchify(X_train, 100):
  X_train_trans.append(ae.transform(x_b))
X_train_trans = np.concatenate(X_train_trans)

print("Transforming test data")
X_test_trans = []
for x_b in batchify(X_test, 100):
  X_test_trans.append(ae.transform(x_b))
X_test_trans = np.concatenate(X_test_trans)

print("Fitting svm")
svm = LinearSVC()
svm.fit(X_train_trans, y_train[:len(X_train_trans)])
print(svm.score(X_test_trans, y_test[:len(X_test_trans)]))