import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.python.training.tracking.tracking import Asset
from dataset import load_data,read_image,show_example,read_mask,convert2TfDataset
from config import *
from models.Unet import build_unet,compile_unet
from models.Unetmobilev2 import build_unet_mb2,compile_unet_mb2
# from models.r2attentionunet import build_unet_v2,build_r2_unet,build_att_unet,build_att_r2_unet,compile_unet_v2,compile_r2_unet,compile_att_r2_unet,compile_att_unet
from models.Attentionunet import build_att_unet,compile_att_unet
from models.AttentionUnetmobilev2 import build_att_unet_mb2,compile_att_unet_mb2
from models.Unetefficientb0 import build_unet_eff0,compile_unet_eff0
from models.AttentionUnetefficientb0 import build_att_unet_eff0,compile_att_unet_eff0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

images, masks = load_data(IMAGE_PATH,MASK_PATH)
print(f'Amount of images: {len(images)}')

###
# Show image examples 
# image, mask = read_image(images[123]), read_mask(masks[123])
# show_example(image, mask)
###

# Separate image paths into training_set, validdation_set
train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)

print(f'Training: {len(train_x)} - Validation: {len(valid_x)}')


train_step = len(train_x)//BATCH_SIZE
if len(train_x) % BATCH_SIZE != 0:
    train_step += 1

valid_step = len(valid_x)//BATCH_SIZE
if len(valid_x) % BATCH_SIZE != BATCH_SIZE:
    valid_step += 1
    
print(f'{train_step} - {valid_step}')

train_dataset = convert2TfDataset(train_x, train_y, BATCH_SIZE)
valid_dataset = convert2TfDataset(valid_x, valid_y, BATCH_SIZE)



if (MODEL_SELECTION == "unet"):
    # Unet model
    print("Using unet model ")
    unet = build_unet()
    unet.summary()
    unet,callbacks = compile_unet(unet)
    H = unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=20,
                callbacks=callbacks)

elif (MODEL_SELECTION == "unet_eff0"):
    # Unet model (other type of coding)
    print("Using unet_eff0 model ")
    unet_eff0 = build_unet_eff0()
    unet_eff0.summary()
    unet_eff0,callbacks = compile_unet_eff0(unet_eff0)
    H = unet_eff0.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=20,
                callbacks=callbacks)

elif (MODEL_SELECTION == "mb2_unet"):
    #Mobilenetv2 Unet model 
    print("Using mb2_unet model ")
    mobilenetv2_unet = build_unet_mb2()
    mobilenetv2_unet.summary()
    mobilenetv2_unet,callbacks = compile_unet_mb2(mobilenetv2_unet)
    H = mobilenetv2_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=20,
                callbacks=callbacks)
    
elif (MODEL_SELECTION == "att_unet"): 
    print("Using att_unet model ")
    att_unet = build_att_unet()
    att_unet.summary()
    att_unet,callbacks = compile_att_unet(att_unet)
    H = att_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=40,
                callbacks=callbacks)

elif (MODEL_SELECTION == "att_unet_mb2"): 
    print("Using att_unet_mb2 model ")
    att_mobile_unet = build_att_unet_mb2()
    att_mobile_unet.summary()
    att_mobile_unet,callbacks = compile_att_unet_mb2(att_mobile_unet)
    H = att_mobile_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=40,
                callbacks=callbacks)
    
elif (MODEL_SELECTION == "unet_att_eff0"): 
    print("Using unet_att_eff0 model ")
    r2_unet = build_att_unet_eff0()
    r2_unet.summary()
    r2_unet,callbacks = compile_att_unet_eff0(r2_unet)
    H = r2_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=20,
                callbacks=callbacks)

# elif (MODEL_SELECTION == "att_r2_unet"): 
#     print("Using att_r2_unet model ")
#     att_r2_unet = build_att_r2_unet()
#     att_r2_unet.summary()
#     att_r2_unet,callbacks = compile_att_r2_unet(att_r2_unet)
#     H = att_r2_unet.fit(train_dataset,
#                 validation_data=valid_dataset,
#                 steps_per_epoch=train_step,
#                 validation_steps=valid_step,
#                 epochs=20,
#                 callbacks=callbacks)

else:
    raise AssertionError("Model not supported. Currently support unet, unet_v2, mb2_unet, att_unet. r2_unet and att_r2_unet ")

fig = plt.figure()
numOfEpoch = 20
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()