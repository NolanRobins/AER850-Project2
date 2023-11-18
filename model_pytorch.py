import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


while True:
    if(torch.cuda.is_available() == False):
        response = input("\033[93mGPU NOT DETECTED AND WILL NOT BE USED.\nPERFORMANCE MAY BE REDUCED. CONTINUE ANYWAYS?\033[0m \033[96m Y/[N]:\033[0m ")
        if response == 'y' or response == 'Y':
            device = torch.device('cpu')
            break
        elif response == 'n' or response == 'N' or response == '':
            exit()
        else:
            print("Input Not Valid")
    else:
        device = torch.device('cuda:0')
        break


train_transformer = transforms.Compose([
    transforms.RandomAffine(degrees = (-45, 45), translate = (0, 0.3), scale = (0.8, 1.3), shear = 45),
    transforms.ColorJitter(brightness = 0.4, contrast = 0.15),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

rescaler = transforms.Compose(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

train_set = datasets.ImageFolder('Data/Test', transform = train_transformer)

print(train_set.class_to_idx)
classes = train_set.class_to_idx
classes = list(k for k, _ in classes.items())

train_set = DataLoader(train_set, batch_size = 16, shuffle = True)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_set)
images, labels = next(dataiter)



imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))



# val_set = tf.keras.utils.image_dataset_from_directory(
#     "Data/Validation",
#     labels="inferred",
#     label_mode="categorical",
#     image_size=(100,100),
#     color_mode="rgb",
# )

# train_set = tf.keras.utils.image_dataset_from_directory(
#     "Data/Train",
#     labels="inferred",
#     label_mode="categorical",
#     image_size=(100,100),
#     color_mode="rgb",
# )


# plt.figure(figsize=(10, 10))
# for images, labels in train_set.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.axis("off")

# plt.show()



# print(train_set.class_names)

# AUTOTUNE = tf.data.AUTOTUNE

# train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


# model = keras.Sequential([
#     layers.RandomRotation((-0.4,0.4), seed = 727),
#     layers.RandomTranslation((-0.2,0.2), (-0.2,0.2), seed = 727),
#     layers.RandomBrightness((-0.5,0.5), seed = 727),
#     # layers.RandomZoom((-0.2,0.2), seed = 727),

#     layers.Rescaling(1./255),

#     layers.Conv2D(16, 3, activation= 'relu'),
#     layers.MaxPooling2D(),

#     layers.Conv2D(32, 3, activation= 'relu'),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64, 3, activation= 'relu'),
#     layers.MaxPooling2D(),

#     layers.Flatten(),
#     # layers.Dense(128, activation="relu"),
#     layers.Dropout(0.5),
#     layers.Dense(4, activation='softmax'),
# ])

# model.compile(optimizer=keras.optimizers.Adam(),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])





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