import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras_cv
import PIL

BATCH_SIZE = 32

while True:
    if(tf.config.list_physical_devices('GPU') == []):
        response = input("\033[93mGPU NOT DETECTED AND WILL NOT BE USED.\nPERFORMANCE MAY BE REDUCED. CONTINUE ANYWAYS?\033[0m \033[96m Y/[N]:\033[0m ")
        if response == 'y' or response == 'Y':
            break
        elif response == 'n' or response == 'N' or response == '':
            exit()
        else:
            print("Input Not Valid")
    else:
        break

train_preprocessor = tf.keras.Sequential([
    # keras_cv.layers.RandomCropAndResize(target_size=(200, 200), crop_area_factor=(0, 1.0), aspect_ratio_factor=(0.9, 1.1), seed = 852),
    layers.RandomZoom((-0.5, 0.5), (-0.5, 0.5), seed = 852),
    layers.RandomTranslation(0, (-0.2,0.2), seed = 852),
    keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, seed = 852),
    # layers.RandomRotation((-0.4,0.4), seed = 852),
    

    # layers.RandomBrightness((0,0.5), seed = 852),
    keras_cv.layers.RandomSharpness((0.8, 0.8), (0, 255), seed = 852),
    # keras_cv.layers.Equalization((0, 255), 256),
    keras_cv.layers.AutoContrast((0, 255)),
    layers.Resizing(100, 100),
    layers.Rescaling(1./255),
])

val_preprocessor = tf.keras.Sequential([
    layers.Rescaling(1./255),
])

train_set = tf.keras.utils.image_dataset_from_directory(
    "Data/Train",
    labels="inferred",
    label_mode="categorical",
    image_size=(300, 300),
    color_mode="rgb",
    batch_size = BATCH_SIZE,
    seed = 852,
    # interpolation = "nearest",
)

class_names = train_set.class_names

train_set = train_set.map(lambda x, y: (train_preprocessor(x, training = True), y))

val_set = tf.keras.utils.image_dataset_from_directory(
    "Data/Validation",
    labels="inferred",
    label_mode="categorical",
    image_size=(100,100),
    color_mode="rgb",
    batch_size = BATCH_SIZE,
    seed = 852,
)

val_set = val_set.map(lambda x, y: (val_preprocessor(x), y))


for images, labels in train_set.take(1):
    augmented_images = images.numpy()


plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis("off")
plt.savefig("ModelOutput/datafig.png")


for images, labels in val_set.take(1):
    augmented_images = images.numpy()

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis("off")
plt.savefig("ModelOutput/valfig.png")


AUTOTUNE = tf.data.AUTOTUNE

train_set = train_set.cache().shuffle(1600).prefetch(buffer_size=AUTOTUNE)


model = keras.Sequential([
    layers.Conv2D(256, 3, strides=(1, 1), activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation= 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),

    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(),
    

    layers.Flatten(),
    # layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

optimizer=keras.optimizers.Adam()
optimizer.learning_rate.assign(0.0003)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.build((BATCH_SIZE, 100, 100, 3))

model.summary()

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001)

epochs = 250
history = model.fit(
  train_set,
  validation_data=val_set,
  epochs=epochs,
#   callbacks = reduce_lr,
)

model.save("ModelOutput/modeled.keras")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("ModelOutput/history.png")

validation_accuracy = model.evaluate(val_set)

# Print the validation accuracy
print("Validation Accuracy:", validation_accuracy)

print(model.evaluate(val_set, verbose = 1))
