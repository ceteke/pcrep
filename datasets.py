import pickle, joblib, os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class ShapeNet(object):
  def __init__(self, base_dir, num_batch):
    self.base_dir = base_dir
    self.num_batch = num_batch

  def process(self):
    dataset = []
    for i in range(self.num_batch):
      dataset.append(self.load_batch(i))
    dataset = np.concatenate(dataset)
    return dataset.reshape(-1, dataset.shape[1], 3)

  def load_batch(self, i):
    return pickle.load(open(os.path.join(self.base_dir, 'shapenet_pc_{}.pk'.format(i)), 'rb'))

class Tork18(object):
  def __init__(self, base_dir, skill):
    self.base_dir = base_dir
    self.skill = skill

  def get_dataset(self):
    skill_points_dir = os.path.join(self.base_dir, '{}_points.pk'.format(self.skill))
    skill_labels_dir = os.path.join(self.base_dir, '{}_labels.pk'.format(self.skill))

    skill_points = pickle.load(open(skill_points_dir, 'rb'))
    skill_labels = pickle.load(open(skill_labels_dir, 'rb'))

    X = np.array(skill_points)
    y = np.array(skill_labels)

    return X, y