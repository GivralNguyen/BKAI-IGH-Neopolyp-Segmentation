import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from dataset import load_data,read_image,show_example,read_mask,convert2TfDataset
from config import *
from models.Unet import build_unet,compile_unet
from models.Unetmobilev2 import build_unet_mb2,compile_unet_mb2
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

### 
# Unet model
model = build_unet()
model.summary()
model,callbacks = compile_unet(model)
H = model.fit(train_dataset,
             validation_data=valid_dataset,
             steps_per_epoch=train_step,
             validation_steps=valid_step,
             epochs=20,
             callbacks=callbacks)
### 

### 
# Mobilenetv2 Unet model 
# mobilenetv2_unet = build_unet_mb2()
# mobilenetv2_unet.summary()
# mobilenetv2_unet,callbacks = compile_unet_mb2(mobilenetv2_unet)
# H = mobilenetv2_unet.fit(train_dataset,
#              validation_data=valid_dataset,
#              steps_per_epoch=train_step,
#              validation_steps=valid_step,
#              epochs=10,
#              callbacks=callbacks)
###



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