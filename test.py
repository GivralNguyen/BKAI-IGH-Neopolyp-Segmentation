from tensorflow import keras
from tqdm import tqdm
import os 
import numpy as np
from dataset import read_image,show_example,read_image_ori
from config import SAVE_FOLDER,TEST_PATH,WEIGHT_PATH
import imageio
import cv2 
model = keras.models.load_model(WEIGHT_PATH)
save_folder = SAVE_FOLDER
test_images = [os.path.join(TEST_PATH, f'{x}') for x in os.listdir(
TEST_PATH)]
for image in tqdm(test_images):
    save_name = image.split('/')[-1]
    # print(save_name)
    original_image = cv2.imread(image)
    h, w, c = original_image.shape
    x = read_image(image)
    p = model.predict(np.expand_dims(x, axis=0))[0]
    p = np.argmax(p, axis=-1)
    # p = cv2.resize(p.astype(np.float32), ( w , h ), cv2.INTER_NEAREST) 
    # x = read_image_ori(image)
    # x = x*255 # Get back image pixels
    # imageio.imwrite(save_folder+'/'+save_name, p)
    show_example(x, p)