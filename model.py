import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

while True:
    if(tf.config.list_physical_devices('GPU') == []):
        response = input("\033[93mGPU NOT DETECTED AND WILL NOT BE USED.\nPERFORMANCE MAY BE REDUCED. CONTINUE ANYWAYS?\033[0m \033[96m Y/[N]:\033[0m ")
        if response == 'y' or response == 'Y':
            break
        elif response == 'n' or response == 'N' or response == '':
            exit()
        else:
            print("Input Not Valid")
        #print("\n")
    else:
        break


train_set = tf.keras.utils.image_dataset_from_directory(
    "Data\Train",
    labels="inferred",
    label_mode="categorical",
    image_size=(100,100)
)