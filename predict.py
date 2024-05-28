import sys
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image

class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50Custom, self).__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Get the number of features in the last layer
        num_ftrs = self.base_model.fc.in_features
        # Replace the last layer with a new layer that matches the number of classes
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(num_ftrs, num_classes)  # Final layer for our number of classes
        )

    def forward(self, x):
        # Define the forward pass
        return self.base_model(x)

# Function to load the trained model from a file
def load_model(model_path, device):
    model = ResNet50Custom(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict the class of a spectogran
def predict_image(model, image_path, transform, device):
    # Open the image file
    image = Image.open(image_path)
    # Apply the transformations to the image
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():  # No need to track gradients for prediction
        output = model(image)
        _, prediction = torch.max(output, 1)  # Get the index of the max log-probability
        class_name = idx_to_class[prediction.item()]  # Convert index to class name
        
    return class_name

if __name__ == "__main__":
    # Ensure the script was called with the correct arguments
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <image_path>")
        sys.exit(1)

    # Parse arguments
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Define the transformations to apply to the input image
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the input image
        transforms.CenterCrop(224),  # Crop to the size expected by ResNet
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    # Load the class index mapping from the dataset
    dataset_dir = 'subsampled_data'
    dataset = datasets.ImageFolder(root=dataset_dir)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Load the model and predict the class of the image
    model = load_model(model_path, device)
    predicted_class = predict_image(model, image_path, transform, device)
    
    # Print the predicted class
    print(f"Predicted class: {predicted_class}")
