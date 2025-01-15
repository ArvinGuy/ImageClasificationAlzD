import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model
model = torchvision.models.resnet18(pretrained=True).to(device)

# Modify the final layer for 4 classes
num_classes = 4
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes).to(device)

# Load your trained model weights
# Replace 'model_weights.pth' with your actual trained weights file
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

# Define image transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Streamlit app interface
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the following categories:")
st.write(classes)

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    transformed = transformations(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predicted_output = model(transformed)
        _, predicted = torch.max(predicted_output, 1)
        predicted_class = classes[predicted.item()]

    # Display prediction
    st.write("### Predicted Class:")
    st.write(predicted_class)

# Visualization of training metrics
if st.checkbox("Show Training Metrics (Confusion Matrix, Accuracy, Loss)"):
    # Replace the following with your actual metrics if needed
    # These are placeholders for demonstration purposes

    # Example confusion matrix
    st.write("#### Confusion Matrix")
    dummy_cf_matrix = np.array([[50, 2, 3, 1], [4, 45, 0, 1], [1, 0, 48, 2], [2, 1, 0, 47]])
    st.write(dummy_cf_matrix)

    # Example classification report
    st.write("#### Classification Report")
    dummy_report = """\
              precision    recall  f1-score   support

    Mild Impairment       0.91      0.90      0.90        56
 Moderate Impairment       0.88      0.91      0.89        50
       No Impairment       0.96      0.92      0.94        52
Very Mild Impairment       0.91      0.93      0.92        50

       accuracy                           0.91       208
      macro avg       0.91      0.91      0.91       208
   weighted avg       0.91      0.91      0.91       208
    """
    st.text(dummy_report)

    # Example loss graph
    st.write("#### Training Loss")
    dummy_training_loss = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
    plt.plot(range(1, len(dummy_training_loss) + 1), dummy_training_loss, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    st.pyplot(plt)

    # Example accuracy graph
    st.write("#### Training Accuracy")
    dummy_training_accuracy = [50, 65, 72, 80, 85, 91]
    plt.plot(range(1, len(dummy_training_accuracy) + 1), dummy_training_accuracy, marker='o', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    st.pyplot(plt)
