import torch
import torch.nn as nn
from torchvision import models

def get_fer_model(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
if __name__ == "__main__":
    print("Instanciando el modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")
    model = get_fer_model(num_classes=7).to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    outputs = model(dummy_input)
    print(f"entry : {dummy_input.shape}")
    print(f"output : {outputs.shape}")
    print("successfully instantiated the model and performed a forward pass with dummy data.")