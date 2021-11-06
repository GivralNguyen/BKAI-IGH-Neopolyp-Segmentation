# from tensorflow import keras 
# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
# from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,add, multiply
# from keras.layers import concatenate, core, Dropout
# from keras.models import Model
# from keras.layers.merge import concatenate
# from keras.optimizers import Adam
# from keras.optimizers import SGD
# from keras.layers.core import Lambda
# import keras.backend as K
# import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# def up_and_concate(down_layer, layer):

#     in_channel = down_layer.get_shape().as_list()[3]

#     # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
#     up = UpSampling2D(size=(2, 2))(down_layer)

    
#     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

#     concate = my_concat([up, layer])

#     return concate


# def attention_up_and_concate(down_layer, layer):
    
#     in_channel = down_layer.get_shape().as_list()[3]

#     # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
#     up = UpSampling2D(size=(2, 2))(down_layer)

#     layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)

    
#     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

#     concate = my_concat([up, layer])
#     return concate


# def attention_block_2d(x, g, inter_channel):
#     # theta_x(?,g_height,g_width,inter_channel)

#     theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

#     # phi_g(?,g_height,g_width,inter_channel)

#     phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

#     # f(?,g_height,g_width,inter_channel)

#     f = Activation('relu')(add([theta_x, phi_g]))

#     # psi_f(?,g_height,g_width,1)

#     psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

#     rate = Activation('sigmoid')(psi_f)

#     # rate(?,x_height,x_width)

#     # att_x(?,x_height,x_width,x_channel)

#     att_x = multiply([x, rate])

#     return att_x


# def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

#               padding='same', data_format='channels_last'):
#     if data_format == 'channels_first':
#         input_n_filters = input_layer.get_shape().as_list()[1]
#     else:
#         input_n_filters = input_layer.get_shape().as_list()[3]

#     layer = input_layer
#     for i in range(2):
#         layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
#         if batch_normalization:
#             layer = BatchNormalization()(layer)
#         layer = Activation('relu')(layer)
#         layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
#         layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

#     if out_n_filters != input_n_filters:
#         skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
#             input_layer)
#     else:
#         skip_layer = input_layer
#     out_layer = add([layer, skip_layer])
#     return out_layer


# # Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
# def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

#                   padding='same', data_format='channels_last'):
#     if data_format == 'channels_first':
#         input_n_filters = input_layer.get_shape().as_list()[1]
#     else:
#         input_n_filters = input_layer.get_shape().as_list()[3]

#     if out_n_filters != input_n_filters:
#         skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
#             input_layer)
#     else:
#         skip_layer = input_layer

#     layer = skip_layer
#     for j in range(2):

#         for i in range(2):
#             if i == 0:

#                 layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
#                     layer)
#                 if batch_normalization:
#                     layer1 = BatchNormalization()(layer1)
#                 layer1 = Activation('relu')(layer1)
#             layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
#                 add([layer1, layer]))
#             if batch_normalization:
#                 layer1 = BatchNormalization()(layer1)
#             layer1 = Activation('relu')(layer1)
#         layer = layer1

#     out_layer = add([layer, skip_layer])
#     return out_layer

# ########################################################################################################
# # Define the neural network
# def build_unet_v2(shape=(256,256,3)):
#     inputs = Input(shape=shape) #256 256 3
#     x = inputs #256 256 3
#     depth = 4
#     features = 64
#     skips = []
#     for i in range(depth):
#         x = Conv2D(features, (3, 3), padding='same')(x) # #256 256 64 / 128 128 128 / 64 64 256 /32 32 512 
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         # x = Dropout(0.2)(x) # #256 256 64 / 128 128 128  / 64 64 256 /32 32 512
#         x = Conv2D(features, (3, 3), padding='same')(x) # #256 256 64 / / 128 128 128 / 64 64 256 /32 32 512 
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         skips.append(x) # 256 256 64 / 128 128 128  / 64 64 256 /32 32 512 
#         x = MaxPooling2D((2, 2))(x) # 128 128 64 / / 64 64 128 / 32 32 256   / 16 16 512 
#         features = features * 2 #128 256 512 1024

#     x = Conv2D(features, (3, 3), padding='same')(x) # 16 16 1024 
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(features, (3, 3), padding='same')(x) # 16 16 1024  
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     for i in reversed(range(depth)):
#         features = features // 2
#         # attention_up_and_concate(x,[skips[i])
#         x = Conv2DTranspose(features, (2,2), strides=2, padding='same')(x)
#         # x = UpSampling2D(size=(2, 2))(x)
#         x = Concatenate()([x, skips[i]])
#         x = Conv2D(features, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(features, (3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#     output = Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='softmax')(x) # 256x256x3
#     model = Model(inputs=inputs, outputs=output)

#     #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
#     return model


# ########################################################################################################
# #Attention U-Net
# def build_att_unet(shape=(256,256,3)):
#     inputs = Input(shape=shape)
#     x = inputs
#     depth = 4
#     features = 64
#     skips = []
#     for i in range(depth):
#         x = Conv2D(features, (3, 3), padding='same')(x) # #256 256 64 / 128 128 128 / 64 64 256 /32 32 512 
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         # x = Dropout(0.2)(x) # #256 256 64 / 128 128 128  / 64 64 256 /32 32 512
#         x = Conv2D(features, (3, 3), padding='same')(x) # #256 256 64 / / 128 128 128 / 64 64 256 /32 32 512 
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         skips.append(x) # 256 256 64 / 128 128 128  / 64 64 256 /32 32 512 
#         x = MaxPooling2D((2, 2))(x) # 128 128 64 / / 64 64 128 / 32 32 256   / 16 16 512 
#         features = features * 2 #128 256 512 1024

#     x = Conv2D(features, (3, 3), padding='same')(x) # 16 16 1024 
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(features, (3, 3), padding='same')(x) # 16 16 1024  
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     for i in reversed(range(depth)):
#         features = features // 2
#         # x is down_layer, skips[i] is layer
#         # x = attention_up_and_concate(x, skips[i])
#         in_channel = x.get_shape().as_list()[3]

#         # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
#         up = UpSampling2D(size=(2, 2))(x)

#         layer = attention_block_2d(x=skips[i], g=up, inter_channel=in_channel // 4)

        
#         my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

#         x = my_concat([up, layer])
#         x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

#     output = Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='softmax')(x) # 256x256x3
#     model = Model(inputs=inputs, outputs=output)

#     #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
#     return model


# ########################################################################################################
# #Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
# def build_r2_unet(shape=(256,256,3)):
#     inputs = Input(shape=shape)
#     x = inputs
#     depth = 4
#     features = 64
#     skips = []
#     for i in range(depth):
#         x = rec_res_block(x, features)
#         skips.append(x)
#         x = MaxPooling2D((2, 2))(x)

#         features = features * 2

#     x = rec_res_block(x, features)

#     for i in reversed(range(depth)):
#         features = features // 2
#         x = up_and_concate(x, skips[i])
#         x = rec_res_block(x, features)

#     conv6 = Conv2D(3, (1, 1), padding='same')(x)
#     conv7 = core.Activation('sigmoid')(conv6)
#     model = Model(inputs=inputs, outputs=conv7)
#     #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
#     return model


# ########################################################################################################
# #Attention R2U-Net
# def build_att_r2_unet(shape=(256,256,3)):
#     inputs = Input(shape=shape)
#     x = inputs
#     depth = 4
#     features = 64
#     skips = []
#     for i in range(depth):
#         x = rec_res_block(x, features)
#         skips.append(x)
#         x = MaxPooling2D((2, 2))(x)

#         features = features * 2

#     x = rec_res_block(x, features)

#     for i in reversed(range(depth)):
#         features = features // 2
#         x = attention_up_and_concate(x, skips[i])
#         x = rec_res_block(x, features)

#     conv6 = Conv2D(3, (1, 1), padding='same')(x)
#     conv7 = core.Activation('sigmoid')(conv6)
#     model = Model(inputs=inputs, outputs=conv7)
#     #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
#     return model

# def compile_unet_v2(unet_v2):
#     unet_v2.compile(loss='categorical_crossentropy',
#              optimizer=tf.keras.optimizers.Adam(1e-4),
#              metrics=[
#                  tf.keras.metrics.Precision(),
#                  tf.keras.metrics.Recall(),
#                  tf.keras.metrics.CategoricalAccuracy(name='acc')
#                 #  tf.keras.metrics.MeanIoU(num_classes=3)
#              ])

#     callbacks = [
#         ModelCheckpoint('unet_v2.h5', verbose=1, save_best_model=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
#         EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#     ]
    
#     return unet_v2, callbacks

# def compile_att_unet(att_unet):
#     att_unet.compile(loss='categorical_crossentropy',
#              optimizer=tf.keras.optimizers.Adam(1e-4),
#              metrics=[
#                  tf.keras.metrics.Precision(),
#                  tf.keras.metrics.Recall(),
#                  tf.keras.metrics.CategoricalAccuracy(name='acc')
#                 #  tf.keras.metrics.MeanIoU(num_classes=3)
#              ])

#     callbacks = [
#         ModelCheckpoint('att_unet.h5', verbose=1, save_best_model=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
#         EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#     ]
    
#     return att_unet, callbacks

# def compile_r2_unet(r2_unet):
#     r2_unet.compile(loss='categorical_crossentropy',
#              optimizer=tf.keras.optimizers.Adam(1e-4),
#              metrics=[
#                  tf.keras.metrics.Precision(),
#                  tf.keras.metrics.Recall(),
#                  tf.keras.metrics.CategoricalAccuracy(name='acc')
#                 #  tf.keras.metrics.MeanIoU(num_classes=3)
#              ])

#     callbacks = [
#         ModelCheckpoint('r2_unet.h5', verbose=1, save_best_model=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
#         EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#     ]
    
#     return r2_unet, callbacks

# def compile_att_r2_unet(att_r2_unet):
#     att_r2_unet.compile(loss='categorical_crossentropy',
#              optimizer=tf.keras.optimizers.Adam(1e-4),
#              metrics=[
#                  tf.keras.metrics.Precision(),
#                  tf.keras.metrics.Recall(),
#                  tf.keras.metrics.CategoricalAccuracy(name='acc')
#                 #  tf.keras.metrics.MeanIoU(num_classes=3)
#              ])

#     callbacks = [
#         ModelCheckpoint('att_r2_unet.h5', verbose=1, save_best_model=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
#         EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#     ]
    
#     return att_r2_unet, callbacks