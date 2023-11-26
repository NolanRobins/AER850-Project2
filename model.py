import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras_cv
import PIL
from PIL import Image, ImageDraw, ImageFont
import predict_model

BATCH_SIZE = 32

def main():
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
        layers.RandomTranslation(0, (-0.2,0.2), seed = 852),
        keras_cv.layers.RandomShear(x_factor=0.1, y_factor=0.1, seed = 852),
        layers.GaussianNoise(0.3, seed = 727),
        layers.RandomZoom((-0.15, 0.15), (-0.15, 0.15), fill_mode = "constant", seed = 852),
        # layers.RandomBrightness((0,0.5), seed = 852),
        # keras_cv.layers.RandomSharpness((0.8, 0.8), (0, 255), seed = 852),
        # keras_cv.layers.AutoContrast((0, 255)),
        # layers.Resizing(100, 100),
        layers.Rescaling(1./255),
    ])

    val_preprocessor = tf.keras.Sequential([
        layers.Rescaling(1./255),
    ])

    train_set = tf.keras.utils.image_dataset_from_directory(
        "Data/Train",
        labels="inferred",
        class_names=["Large", "Medium", "Small", "None"],
        label_mode="categorical",
        image_size=(100, 100),
        color_mode="rgb",
        batch_size = BATCH_SIZE,
        seed = 852,
        # interpolation = "nearest",
    )

    print(train_set.class_names)
    labels = train_set.class_names

    train_set = train_set.map(lambda x, y: (train_preprocessor(x, training = True), y))

    val_set = tf.keras.utils.image_dataset_from_directory(
        "Data/Validation",
        labels="inferred",
        label_mode="categorical",
        image_size=(100,100),
        class_names=["Large", "Medium", "Small", "None"],
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
        layers.InputLayer((100,100,3)),

        layers.Conv2D(256, 2, strides=(1, 1), activation= 'relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 2, strides=(1, 1), activation= 'relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),

        layers.Conv2D(64, 4, activation= 'leaky_relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.35),

        layers.Conv2D(64, 3, activation= 'relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),

        layers.Conv2D(64, 3, activation= 'relu'),
        layers.MaxPooling2D(),
    

        layers.Flatten(),
        layers.Dense(256, activation='elu'),
        layers.Dropout(0.5),

        layers.Dense(4, activation='softmax')
    ])

    optimizer=keras.optimizers.Adam()
    optimizer.learning_rate.assign(0.0005)

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    model.build((BATCH_SIZE, 100, 100, 3))

    model.summary()

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001)

    epochs = 60
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

    predict_model.predict_image(model, "Data/Train/Medium/Crack__20180419_05_06_26,199.bmp")
    execute_predict()
    



def execute_predict():
    loaded_model = keras.models.load_model("ModelOutput/modeled.keras")
    predict_model.predict_image(loaded_model, "Data/Test/Medium/Crack__20180419_06_19_09,915.bmp")
    predict_model.predict_image(loaded_model, "Data/Test/Large/Crack__20180419_13_29_14,846.bmp")
    


if __name__ == "__main__":
    main()

