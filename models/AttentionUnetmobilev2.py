import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate,Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,add, multiply
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K
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

def att_expansive_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    a = AttnGatingBlock(skip_features, x,num_filters//2)
    x = Concatenate()([x, a])
    x = conv_block(x, num_filters)
    return x

def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2]),
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1],
                                             shape_x[2] //
                                             shape_sigmoid[2]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=shape_x[3],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output

def build_att_unet_mb2(shape=(256,256,3)):
    """ INPUT """
    inputs = Input(shape=shape, name='input')

    """ BACKBONE MobileNetV2 """
    encoder = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)

    """ Encoder """
    s1 = encoder.get_layer('input').output 
    s2 = encoder.get_layer('block_1_expand_relu').output
    s3 = encoder.get_layer('block_3_expand_relu').output        
    s4 = encoder.get_layer('block_6_expand_relu').output

    """ Bridge """
    b1 = encoder.get_layer('block_13_expand_relu').output        

    """ Decoder """
    d1 = att_expansive_block(b1, s4, 512)
    d2 = att_expansive_block(d1, s3, 256)
    d3 = att_expansive_block(d2, s2, 128)
    d4 = att_expansive_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(3, (1,1), 1, 'same', activation='softmax')(d4)

    return Model(inputs, outputs, name='MobilenetV2_Unet')

def compile_att_unet_mb2(att_unet_mb2):
    att_unet_mb2.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(1e-4),
             metrics=[
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.CategoricalAccuracy(name='acc')
                #  tf.keras.metrics.MeanIoU(num_classes=3)
             ])

    callbacks = [
        ModelCheckpoint('att_unet_mb2.h5', verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]   
    return att_unet_mb2,callbacks