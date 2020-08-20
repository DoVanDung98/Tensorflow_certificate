import tesnorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

class myCallback(tf.keras.callback.Callback):
    def on_epoch_end(self, epoch, log ={}):
        if (log.get("loss"))