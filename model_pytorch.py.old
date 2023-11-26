import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

class MainNN(nn.Module):
    def __init__(self):
        super(MainNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 25)  # Adjust the view shape based on your image dimensions
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# model = keras.Sequential([

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

#Defined pytorch's batch sizes
batch_sizes = 4

def main():
    #Check for Nvidia CUDA GPU, if not found raise alert and use CPU if user overrides
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

    #Define the training set's data processing transformer
    train_transformer = transforms.Compose([
        # transforms.RandomCrop(1200), #Randomly zoom into the image from 2000 x 2000 to anywhere 1200 x 1200
        transforms.RandomAffine(degrees = (-45, 45), translate = (0, 0.3), scale = (0.8, 2.3), shear = 45), # randomly rotate by upto 45deg in either direction, translate the image 30% in any direction, 
                                        #randomly zoom into the image, and shear the image up to 45deg in either direction
        transforms.ColorJitter(brightness = 0.4, contrast = 0.15), #Adjust brightness randomly by upto 40%, datasets seem to have similar brightness in each so to make sure it doesn't train on brightness
                                        #randomize it. Contrast also randomized by upto 15%
        transforms.Resize((100, 100)),  #Resize image to 100 x 100 input.
        transforms.ToTensor(), #Rescale
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalize/Rescale to 0 -> 1 with mean and std
    ])

    #Validation transformer with only rescaling
    validation_transfomer = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    train_set = datasets.ImageFolder('Data/Train', transform = train_transformer) #import training dataset and apply train_transformer

    #Get classification to id map to understand what each classification actually is
    print(train_set.class_to_idx)
    classes = train_set.class_to_idx
    classes = list(k for k, _ in classes.items())

    #Load train_set and shuffle
    train_set = DataLoader(train_set, batch_size = batch_sizes, shuffle = True, num_workers = 24)
    
    #Display imported training set data
    # def imshow(img):
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # dataiter = iter(train_set)
    # images, labels = next(dataiter)

    

    # imshow(torchvision.utils.make_grid(images, normalize=True))
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_sizes)))


    #Import validation dataset
    val_set = datasets.ImageFolder('Data/Validation', transform = validation_transfomer)
    val_set = DataLoader(val_set, batch_size = batch_sizes, shuffle = False)

    

    network = MainNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0

        with torch.autograd.profiler.profile(record_shapes = True) as prof:
            for i, data in enumerate(train_set, 0):
                inputs, labels = data
                optimizer.zero_grad()
                print("Checkpoint")
                with torch.autograd.profiler.record_function("model_inference"):
                    outputs = network(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_set)}], Loss: {running_loss / 10:.4f}")
                    running_loss = 0.0
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("Finished Training")


if __name__ == '__main__':
    main()


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