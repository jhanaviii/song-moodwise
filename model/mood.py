import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Define the dataset path
dataset_path = '/Applications/general/vscode/Projects/mood rec/mood_dataset'  # Ensure this path is correct

# Define image size and categories
image_size = (48, 48)
categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Calm']


# Load the dataset
X = []
y = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    label = categories.index(category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)
        X.append(img)
        y.append(label)

X = np.array(X).reshape(-1, 48, 48, 1) / 255.0  # Normalize images and reshape for the model
y = np.array(y)
y_encoded = to_categorical(y, num_classes=len(categories))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
input_shape = (image_size[0], image_size[1], 1)  # Grayscale image shape
num_classes = len(categories)

accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    model = create_model(input_shape, num_classes)

    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])
    print(f'Fold accuracy: {scores[1]}')

print(f'Mean accuracy: {np.mean(accuracies)}')
print(f'Standard deviation: {np.std(accuracies)}')

# Save the final model
model.save('model/mood_model.h5')
