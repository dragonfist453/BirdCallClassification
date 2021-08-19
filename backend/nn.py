import sys
from os import path
import os
import librosa
import librosa.display
import librosa.feature
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def build_network():  
    # Neural Network parameters
    numBirds = 5
    dims = (224,224,3)

    # ResNet model with flatten at the end
    resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=dims)
    output = resnet.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    resnet = tf.keras.Model(resnet.input, output)

    # Freeze all layers of ResNet to use the features
    for layer in resnet.layers:
        layer.trainable = False

    # Define fully connected network below resnet
    model = tf.keras.Sequential()
    model.add(resnet)
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(numBirds, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.load_weights(os.getcwd() + '\\backend\\BirdCallClassification.hdf5')

    return model

def audioToMel(file, dataPath, imgPath, overwrite = False):
  filePath = dataPath + file
  outPath = imgPath + file[:-4] + '.png'
  if not path.exists(filePath):
    raise FileNotFoundError
  if not overwrite:
    if path.exists(outPath):
      return True
  y, sr = librosa.load(filePath, duration=10)
  S = librosa.feature.melspectrogram(y=y, sr=sr)
  S_dB = librosa.power_to_db(S**2, ref=np.max)
  plt.axis('off')
  ax = plt.gca()
  ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
  img = librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
  plt.savefig(outPath, bbox_inches='tight', transparent=True, pad_inches=0)
  return True