import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import get_fer_model

def train_model(num_epochs=15, learning_rate=0.001):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 2. Load Data and Model
    print("[*] Loading datasets...")
    train_loader, val_loader, classes = get_dataloaders(batch_size=64)
    
    print("[*] Initializing model...")
    model = get_fer_model(num_classes=len(classes)).to(device)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("[*] Starting training loop...")
    # 4. Main Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # --- NUEVO: Imprimir el progreso cada 100 batches ---
            if (i + 1) % 100 == 0:
                print(f"   -> Procesando batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # 5. Validation Phase
        model.eval()
        correct = 0
        total = 0
        
        # Disable gradient calculation for validation to save memory and compute
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Training Loss: {epoch_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

    # 6. Save the final model weights
    save_path = 'fer_resnet18.pth'
    torch.save(model.state_dict(), save_path)
    print(f"[*] Training complete. Model weights saved to: {save_path}")

if __name__ == "__main__":
    # You can adjust num_epochs based on your hardware capabilities
    train_model(num_epochs=15)