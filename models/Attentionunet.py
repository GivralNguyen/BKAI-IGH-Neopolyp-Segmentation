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

def build_att_unet(shape=(256,256,3)):
    input = Input(shape=shape)
    
    # Contracting path
    s1, p1 = contracting_block(input, 64) # 256x256x64, 128x128x64
    s2, p2 = contracting_block(p1, 128) # 128x128x128, 64x64x128
    s3, p3 = contracting_block(p2, 256) # 64x64x256, 32x32x256
    s4, p4 = contracting_block(p3, 512) # 32x32x512, 16x16x512
    
    # Bridge
    b = conv_block(p4, 1024) # 16x16x1024
    
    # Expansize path 
    e1 = att_expansive_block(b, s4, 512) # 32x32x512
    e2 = att_expansive_block(e1, s3, 256) # 64x64x256
    e3 = att_expansive_block(e2, s2, 128) # 128x128x128
    e4 = att_expansive_block(e3, s1, 64) # 256x256x64
    
    output = Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='softmax')(e4) # 256x256x3
    
    return Model(input, output, name='U-Net')

def compile_att_unet(model):
    model.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(1e-4),
             metrics=[
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.CategoricalAccuracy(name='acc')
                #  tf.keras.metrics.MeanIoU(num_classes=3)
             ])

    callbacks = [
        ModelCheckpoint('att_unet.h5', verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]
    
    return model, callbacks
