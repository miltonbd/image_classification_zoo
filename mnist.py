from __future__ import print_function, absolute_import, division, unicode_literals
from absl import app
from absl import flags
from tqdm import tqdm
import numpy as np

import tensorflow
import tensorflow as tf
import os
print("Tensorflow Version:{}".format(tf.version))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from arguments import *

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

mnist = tf.keras.datasets.mnist


"""
todo: tensorboard logging
"""

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 100, 'Epochs.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# train will train the network, validate, and test
# test will load saved model and show test metrcis
# eval will be the output in deployment time
flags.DEFINE_enum('task', 'train', ['train', 'test', 'eval'], 'Tasks')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')
flags.DEFINE_integer('batch', 10, 'Batch')

flags.DEFINE_string('saved_model', "saved_model/{}_model".format(model_name),'saved model')
flags.DEFINE_string('tflite_model', "saved_model/{}_model.tflite".format(model_name),'tflite model')


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32,3, activation='relu', input_shape=(None, 28,28,1))
    self.flatten = tf.keras.layers.Flatten() # flatten makes only one dimention in all channels
    self.fc1 = tf.keras.layers.Dense(100)
    self.fc2 = tf.keras.layers.Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)

    return x


def load_dataset():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Add a channels dimension
  x_train = x_train[..., None]
  x_test = x_test[..., None]
  x_train=x_train.astype(np.float32)
  x_test=x_test.astype(np.float32)

  batch = FLAGS.batch
  print(x_train.shape)
  print(x_train.dtype)

  train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)
  return train_ds, test_ds

def preprocess():
  pass

def get_loss_function():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  print("Using SparseCategoricalCrossentropy")
  return loss_object

def get_optimizer():
  lr = 0.00001
  momemntum = 0.9
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momemntum)
  print("Using SGD Optimizer with lr:{}, momentum:{}".format(lr, momemntum))

  return optimizer



def build_model():
  # mymodel = MyModel()
  # model=tf.keras.models.Sequential()
  # model.summary()
  # model.add(mymodel)
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  return model

model = build_model()
loss_object = get_loss_function()
optimizer = get_optimizer()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(train_obj):
  images, labels = train_obj
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



import sys

def progress_bar(progress, count ,message):
  sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))


def main(argv):
  if FLAGS.debug:
    print('non-flag arguments:', argv)
  Epochs = FLAGS.epochs
  train_ds, test_ds = load_dataset()

  for epoch in tqdm(range(Epochs)):
    saved_model_path=FLAGS.saved_model
    tf.saved_model.save(model,saved_model_path)
    exit(1)
    for train_obj in train_ds:
      train_step(train_obj)
      msg="Train loss:{}, Accuracy:{}".format(train_loss.result(),train_accuracy.result())
      progress_bar(epoch, Epochs, msg)

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}%, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))



if __name__ == '__main__':
    app.run(main)
