from src.model import TrafficSignCNN 
import torch
from torchvision import transforms
from PIL import Image
from src.model import TrafficSignCNN  # Make sure path is correct

NUM_CLASSES = 43  # Update this to match your dataset

# Load the model
model = TrafficSignCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("C:/Users/prath/OneDrive/Desktop/New folder/outputs/best_model.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Example usage
image_path = r"C:\Users\prath\Downloads\Road_sign\Test\12624.png"  # Replace with your test image path
prediction = predict_image(image_path)
print(f"Predicted class index: {prediction}")
