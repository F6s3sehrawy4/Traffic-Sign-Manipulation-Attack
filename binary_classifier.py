import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Paths to the data folders
real_images_path = "dataset/train/real"
fake_images_path = "dataset/train/fake"

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 18

# Data transformations (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((450, 800)),
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Create datasets and dataloaders
dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
class_names = dataset.classes  # ['fake', 'real']
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a binary classifier model (pretrained ResNet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),  # Single output for binary classification
    nn.Sigmoid()
)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()  # Labels as float for BCELoss

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()  # Remove extra dimension for BCELoss

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            preds = (outputs > 0.5).float()  # Predictions (threshold 0.5)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")


# Train the model
train(model, dataloader, criterion, optimizer, num_epochs)

# Save the model
torch.save(model.state_dict(), "binary_classifier.pth")