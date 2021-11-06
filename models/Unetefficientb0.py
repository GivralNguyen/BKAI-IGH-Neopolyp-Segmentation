import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0,MobileNetV2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def conv_block(input, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def contracting_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D(pool_size=(2,2), strides=2)(x)
    return x, p

def expansive_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet_eff0(shape=(256,256,3)):
    """ INPUT """
    inputs = Input(shape=shape, name='input')

    """ BACKBONE MobileNetV2 """
    encoder = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)

    """ Encoder """
    s1 = encoder.get_layer('input').output # [(None, 256, 256, 3)
    s2 = encoder.get_layer('block2a_expand_activation').output # None, 128, 128, 144 
    s3 = encoder.get_layer('block3a_expand_activation').output # None, 64, 64, 192      
    s4 = encoder.get_layer('block4a_expand_activation').output # None, 32, 32, 288

    """ Bridge """
    b1 = encoder.get_layer('block6a_expand_activation').output        

    """ Decoder """
    d1 = expansive_block(b1, s4, 512)
    d2 = expansive_block(d1, s3, 256)
    d3 = expansive_block(d2, s2, 128)
    d4 = expansive_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(3, (1,1), 1, 'same', activation=  'softmax')(d4)

    return Model(inputs, outputs, name='EfficientB0-Unet')

def compile_unet_eff0(unet_eff0):
    unet_eff0.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(1e-4),
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
    return unet_eff0,callbacks

# shape=(256,256,3)
# inputs = Input(shape=shape, name='input')
# # encoder = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
# encoder = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
# print(encoder.summary())

# """ Encoder """
# s1 = encoder.get_layer('input').output # [(None, 256, 256, 3)
# s2 = encoder.get_layer('block2a_expand_activation').output # None, 128, 128, 144 
# s3 = encoder.get_layer('block3a_expand_activation').output # None, 64, 64, 192      
# s4 = encoder.get_layer('block4a_expand_activation').output # None, 32, 32, 288
# b1 = encoder.get_layer('block6a_expand_activation').output         
