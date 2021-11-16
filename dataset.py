import os 
import cv2 as cv 
import numpy as np 
from config import * 
import matplotlib.pyplot as plt
import tensorflow as tf 

def load_data(IMAGE_PATH,MASK_PATH):
    images = [os.path.join(IMAGE_PATH, f'{x}') for x in os.listdir(IMAGE_PATH)]
    masks = [os.path.join(MASK_PATH, f'{x}') for x in os.listdir(MASK_PATH)]
    return images, masks

def read_image(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.resize(image, (WIDTH, HEIGHT))
    image = image/255.0
    image = image.astype(np.float32)
    return image
def read_image_ori(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    # image = cv.resize(image, (WIDTH, HEIGHT))
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(mask_path):
    image = cv.imread(mask_path)
    image = cv.resize(image, (WIDTH, HEIGHT))
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv.inRange(image, lower1, upper1)
    upper_mask = cv.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask;
    red_mask[red_mask != 0] = 2
    
    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv.inRange(image, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 1
    
    full_mask = cv.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask

def show_example(image, mask):
    image = image*255 # Get back image pixels
    plt.figure(figsize=(18,15))
    plt.subplot(1,3,1)
    plt.imshow(image[...,1], cmap='bone')
    plt.axis('off')
    plt.title('Image')

    plt.subplot(1,3,2)
    plt.imshow(mask, cmap='nipy_spectral')
    plt.axis('off')
    plt.title('Mask')

    plt.subplot(1,3,3)
    plt.imshow(image[...,1], cmap='bone')
    plt.imshow(mask, alpha=0.5, cmap='nipy_spectral')
    plt.axis('off')
    plt.title('Overlay')

    plt.show()
    
# Convert numpy data to tensorflow data
'''
We need segent 3 classes:
    + 0 if the pixel is part of the image background (denoted by black color);
    + 1 if the pixel is part of a non-neoplastic polyp (denoted by green color);
    + 2 if the pixel is part of a neoplastic polyp (denoted by red color).
'''
def convert2TfDataset(x, y, batch_size=8):
    def preprocess(image_path, mask_path):
        def f(image_path, mask_path):
            image_path = image_path.decode()
            mask_path = mask_path.decode()
            image = read_image(image_path)
            mask = read_mask(mask_path)
            return image, mask

        image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.uint8])
        mask = tf.one_hot(mask, 3, dtype=tf.uint8)
        image.set_shape([HEIGHT, WIDTH, 3])
        mask.set_shape([HEIGHT, WIDTH, 3])
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

# Convert numpy data to tensorflow data
'''
We need segent 3 classes:
    + 0 if the pixel is part of the image background (denoted by black color);
    + 1 if the pixel is part of a non-neoplastic polyp (denoted by green color);
    + 2 if the pixel is part of a neoplastic polyp (denoted by red color).
'''
def convert2TfDataset(x, y, batch_size=8):
    def preprocess(image_path, mask_path):
        def f(image_path, mask_path):
            image_path = image_path.decode()
            mask_path = mask_path.decode()
            image = read_image(image_path)
            mask = read_mask(mask_path)
            return image, mask

        image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.uint8])
        mask = tf.one_hot(mask, 3, dtype=tf.uint8)
        image.set_shape([HEIGHT, WIDTH, 3])
        mask.set_shape([HEIGHT, WIDTH, 3])
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset