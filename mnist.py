from __future__ import print_function, absolute_import, division, unicode_literals
from absl import app
from absl import flags
from tqdm import tqdm

import tensorflow
import tensorflow as tf

print(tf.version)

mnist = tf.keras.datasets.mnist
a="data_loader"
import importlib
importlib.import_module(a)


"""
todo: tensorboard logging
"""

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_integer('epochs', 100, 'Epochs.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# train will train the network, validate, and test
# test will load saved model and show test metrcis
# eval will be the output in deployment time
flags.DEFINE_enum('task', 'train', ['train', 'test', 'eval'], 'Tasks')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')
flags.DEFINE_integer('batch', 10, 'Batch')

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32,3, activation='relu')
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
  batch = FLAGS.batch
  print(x_train.shape)

  train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)
  return train_ds, test_ds

def preprocess():
  pass


def build_model():
  model = MyModel()
  return model


def main(argv):
  if FLAGS.debug:
    print('non-flag arguments:', argv)
  Epochs = FLAGS.epochs
  train_ds, test_ds = load_dataset()
  model = build_model()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  @tf.function
  def train_step(images, labels):
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

  for epoch in tqdm(range(Epochs)):
    for images, labels in train_ds:
      train_step(images, labels)
      print("Train loss:{}, Accuracy:{}".format(train_loss.result(),train_accuracy.result()))


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
