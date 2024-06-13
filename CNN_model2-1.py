import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Paths
training_folder = r'C:\Users\user\Desktop\college\DL\final_proj\train_images'
testing_folder = r'C:\Users\user\Desktop\college\DL\final_proj\test_images'

# Image dimensions
img_height, img_width = 256, 256  # Adjust according to your images

# Prepare data
def load_data(target_path):
    images = []
    labels = []

    for filename in os.listdir(target_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            file_path = os.path.join(target_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                # Resize the image to a consistent size
                image = cv2.resize(image, (img_width * 2, img_height))  # Images are concatenated horizontally

                # Split the image into input (LHS) and label (RHS)
                input_img = image[:, :img_width]
                label_img = image[:, img_width:]

                # Determine if there's bleeding based on the label image
                has_bleeding = np.any(label_img > 0)
                label = 1 if has_bleeding else 0

                images.append(input_img)
                labels.append(label)

    images = np.array(images)
    labels = to_categorical(labels, num_classes=2)  # Convert to one-hot encoding

    return images, labels

X_train, y_train = load_data(training_folder)
X_test, y_test = load_data(testing_folder)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    return model

input_shape = (img_height, img_width, 3)  # Adjust based on your image dimensions
model = create_model(input_shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Predict the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Visualize the results
def visualize_results(X, y_true, y_pred, indices):
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[idx])
        plt.title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        plt.axis('off')
    plt.show()

# Choose indices for bleeding and non-bleeding cases
bleeding_indices = [i for i, label in enumerate(y_true_classes) if label == 1][:7]
non_bleeding_indices = [i for i, label in enumerate(y_true_classes) if label == 0][:3]

# Visualize results
visualize_results(X_test, y_true_classes, y_pred_classes, bleeding_indices + non_bleeding_indices)

# Print classification report
print(classification_report(y_true_classes, y_pred_classes))

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bleeding', 'Bleeding'])

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Optionally, you can also visualize other metrics over the training epochs (e.g., accuracy, loss)
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot training history
plot_training_history(history)

# Optional: Save the model
model.save('ct_scan_bleeding_detection_model.h5')