import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Define the classes in the exact order the DataLoader assigned them
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 2. Reconstruct the architecture and load weights
def load_model():
    # Instantiate ResNet18 without pre-trained ImageNet weights
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    # Load your trained weights, forcing CPU execution for the free HF Space tier
    model.load_state_dict(torch.load('fer_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Define the exact same validation transforms used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Prediction function
def predict_emotion(image):
    # Ensure the input is a PIL image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
        
    # Apply transformations and add a batch dimension: [1, 3, 224, 224]
    input_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Return a dictionary formatted for Gradio's Label component
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

# 5. Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Upload an Image or use Webcam"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Emotion"),
    title="Facial Expression Recognition (FER)",
    description="Custom ResNet18 model trained to detect 7 human emotions."
)

if __name__ == "__main__":
    # Launch the web server locally
    interface.launch()