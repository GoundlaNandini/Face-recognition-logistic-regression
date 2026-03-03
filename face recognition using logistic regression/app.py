import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data using OpenCV
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for subfolder in os.listdir(folder):  # Loop over subfolders
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for filename in os.listdir(subfolder_path):  # Loop over files
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)  # Read the image
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize image
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    images.append(gray_img.flatten())  # Flatten the image
                    labels.append(subfolder)  # Use subfolder name as label
    return np.array(images), np.array(labels)

# Step 2: Prepare Data for Training
folder_path = r"D:\project"  # Replace with your dataset path
X, y = load_images_from_folder(folder_path)

# Step 3: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split Data into Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)  # Increase iterations if necessary
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Predict on a New Image
def predict_image(image_path, model, label_encoder):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, (64, 64))  # Resize image to match the training data
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    flat_img = gray_img.flatten()  # Flatten the image
    prediction = model.predict([flat_img])  # Predict using the model
    predicted_label = label_encoder.inverse_transform(prediction)  # Decode the label
    return predicted_label[0]

# Example usage
new_image_path = r"D:\project\class2\vdk.jpg"  # Replace with a valid image path
try:
    predicted_person = predict_image(new_image_path, model, label_encoder)
    print(f"Predicted Person: {predicted_person}")
    
    # Display the image
    img = cv2.imread(new_image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_person}")
    plt.axis("off")
    plt.show()

except ValueError as e:
    print(e)
