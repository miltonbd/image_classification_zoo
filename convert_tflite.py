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

def tflite_convert(model, tflite_model_path):
  # Convert to TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_saved_model(model)
  tflite_model = converter.convert()
  open(tflite_model_path, "wb").write(tflite_model)

if __name__ == "__main__":
    tflite_convert("saved_model/{}_model".format(model_name), "saved_model/{}_model.tflite".format(model_name))