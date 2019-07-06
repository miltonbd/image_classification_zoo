from __future__ import print_function, division, unicode_literals
import sys
import os

saved_model = 'saved_model'
if os.path.isdir( saved_model) and not os.path.exists(saved_model):
    os.makedirs(saved_model)

def progress_bar(progress, count ,message):
  sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))
model_name='mnist'
