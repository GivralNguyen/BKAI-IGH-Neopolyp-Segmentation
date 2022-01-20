import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from config import WIDTH,HEIGHT
import segmentation_models as sm
sm.set_framework('tf.keras')
import keras.layers as layers
import keras.backend as backend
from loss import focalloss,focaltversky
sm.framework()
def build_unet_eff3(shape=(HEIGHT,WIDTH,3)):

    unet_eff3 = sm.Unet('efficientnetb3',input_shape=(HEIGHT, WIDTH, 3), classes=3, activation='sigmoid')
    return unet_eff3
def compile_unet_eff3(unet_eff3):
    unet_eff3.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(2e-5),
             metrics=[
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.CategoricalAccuracy(name='acc')
                #  tf.keras.metrics.MeanIoU(num_classes=3)
             ])

    callbacks = [
        ModelCheckpoint('unet_eff3.h5', verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]   
    return unet_eff3,callbacks


class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

 