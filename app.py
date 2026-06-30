import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN 

# 1. Definir las clases
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 2. Cargar tu modelo ResNet18
def load_emotion_model():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load('fer_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

emotion_model = load_emotion_model()

face_detector = MTCNN(keep_all=False, device='cpu')

# 4. Transformaciones PyTorch
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_emotion(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    boxes, _ = face_detector.detect(image)
    if boxes is not None:
        box = boxes[0]
        face_img = image.crop((box[0], box[1], box[2], box[3]))
    else:
        face_img = image
        
    input_tensor = transform(face_img).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = emotion_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

# 6. Interfaz de Gradio
interfaz = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Sube una foto (cuerpo completo o rostro)"),
    outputs=gr.Label(num_top_classes=3, label="Emoción de la Cara Detectada"),
    title="Reconocimiento Facial Inteligente",
    description="Pipeline de dos etapas: Usa MTCNN para ubicar el rostro en la foto y ResNet18 para clasificar la emoción."
)

if __name__ == "__main__":
    interfaz.launch()