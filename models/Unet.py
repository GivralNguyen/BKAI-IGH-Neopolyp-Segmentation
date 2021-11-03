from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
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

def build_unet(shape=(256,256,3)):
    input = Input(shape=shape)
    
    # Contracting path
    s1, p1 = contracting_block(input, 64) # 256x256x64, 128x128x64
    s2, p2 = contracting_block(p1, 128) # 128x128x128, 64x64x128
    s3, p3 = contracting_block(p2, 256) # 64x64x256, 32x32x256
    s4, p4 = contracting_block(p3, 512) # 32x32x512, 16x16x512
    
    # Bridge
    b = conv_block(p4, 1024) # 16x16x1024
    
    # Expansize path
    e1 = expansive_block(b, s4, 512) # 32x32x512
    e2 = expansive_block(e1, s3, 256) # 64x64x256
    e3 = expansive_block(e2, s2, 128) # 128x128x128
    e4 = expansive_block(e3, s1, 64) # 256x256x64
    
    output = Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='softmax')(e4) # 256x256x3
    
    return Model(input, output, name='U-Net')

def compile_unet(model):
    model.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(1e-4),
             metrics=[
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.CategoricalAccuracy(name='acc')
                #  tf.keras.metrics.MeanIoU(num_classes=3)
             ])

    callbacks = [
        ModelCheckpoint('unet.h5', verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]
    
    return model, callbacks