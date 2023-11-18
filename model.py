import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import PIL



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

val_set = tf.keras.utils.image_dataset_from_directory(
    "Data/Validation",
    labels="inferred",
    label_mode="categorical",
    image_size=(100,100),
    color_mode="rgb",
)

train_set = tf.keras.utils.image_dataset_from_directory(
    "Data/Train",
    labels="inferred",
    label_mode="categorical",
    image_size=(100,100),
    color_mode="rgb",
)

sheared_data = tf.keras.preprocessing.image.random_shear(
    tfds.as_numpy(train_set), 45
)


plt.figure(figsize=(10, 10))
for images, labels in train_set.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")

plt.show()

# for images, labels in train_set.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     label_index = tf.argmax(labels[i])
#     plt.title(train_set.class_names[label_index])
#     plt.axis("off")
# plt.show()


print(train_set.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


model = keras.Sequential([
    layers.RandomRotation((-0.4,0.4), seed = 727),
    layers.RandomTranslation((-0.2,0.2), (-0.2,0.2), seed = 727),
    layers.RandomBrightness((-0.5,0.5), seed = 727),
    # layers.RandomZoom((-0.2,0.2), seed = 727),

    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    # layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# epochs=150
# history = model.fit(
#   train_set,
#   validation_data=val_set,
#   epochs=epochs
# )

# model.summary()


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()