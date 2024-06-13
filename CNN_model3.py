import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import matplotlib.pyplot as plt

# Generator class
class DataGenerator(Sequence):
    def __init__(self, image_filenames, batch_size=16, img_height=256, img_width=256, **kwargs):
        super().__init__(**kwargs)
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_filenames_temp = [self.image_filenames[k] for k in indexes]

        X, y = self.__data_generation(image_filenames_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_filenames))
        np.random.shuffle(self.indexes)

    def __data_generation(self, image_filenames_temp):
        X = np.empty((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)

        for i, img_path in enumerate(image_filenames_temp):
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (self.img_width * 2, self.img_height))
                input_img = image[:, :self.img_width]
                label_img = image[:, self.img_width:]

                input_img = input_img / 255.0
                label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)  # Convert mask to grayscale
                label_img = (label_img > 0).astype(np.float32)  # Ensure binary mask

                X[i,] = input_img
                y[i,] = label_img[..., np.newaxis]  # Expand dims to match shape (256, 256, 1)

        return X, y

# Paths
training_folder = r'C:\Users\user\Desktop\college\DL\final_proj\train_images'
testing_folder = r'C:\Users\user\Desktop\college\DL\final_proj\test_images'

# Get image paths
def get_image_paths(folder):
    image_paths = []
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder, filename))
    return image_paths

train_images = get_image_paths(training_folder)
test_images = get_image_paths(testing_folder)

# Split training data into training and validation sets
train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

# Create data generators
train_generator = DataGenerator(train_images, batch_size=16)
val_generator = DataGenerator(val_images, batch_size=16)
test_generator = DataGenerator(test_images, batch_size=16)

# Define the U-Net Model
def unet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Expansive path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(512, (2, 2), activation='relu', padding='same')(u6)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(256, (2, 2), activation='relu', padding='same')(u7)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(128, (2, 2), activation='relu', padding='same')(u8)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(64, (2, 2), activation='relu', padding='same')(u9)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Define input shape
input_shape = (256, 256, 3)

# Create and compile the model
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=40, validation_data=val_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Optional: Save the model
# model.save('ct_scan_bleeding_segmentation_model.h5')

# Predict on a few test images
predictions = model.predict(test_generator)

# Function to plot the image, ground truth mask, and predicted mask
def plot_sample(image, mask, prediction, idx):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(image[idx], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    plt.imshow(mask[idx].squeeze(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(prediction[idx].squeeze(), cmap='gray')
    plt.show()

# Display the results for 5 random test images
np.random.seed(42)
for i in np.random.randint(0, len(test_generator), 5):
    X_test, y_test = test_generator[i]  # Retrieve a batch of images and masks
    predictions = model.predict(X_test)  # Predict the batch
    plot_sample(X_test, y_test, predictions, 0)  # Plot the first image in the batch