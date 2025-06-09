import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

# Custom Dataset to load images and their true labels
class LabeledImageDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Read the CSV file, specifying the relevant columns
        self.labels_df = pd.read_csv(label_file, usecols=["ImagePath", "Label"])

        # Extract filenames and labels
        self.image_paths = self.labels_df['ImagePath'].values
        self.labels = self.labels_df['Label'].apply(
            lambda x: 1 if x == 'real' else 0).values  # Convert 'real'/'fake' to 1/0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])  # Construct full path
        image = Image.open(image_path).convert('RGB')  # Open image as RGB

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()  # Apply threshold

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# Define the data transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((450, 800)),
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Load the test dataset with the new structure
test_dataset = LabeledImageDataset(
    image_folder=".",
    label_file="sample_submission.csv",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load and evaluate the model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("binary_classifier.pth", weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Evaluate the model
predictions, true_labels = evaluate(model, test_loader, device)

# Compute performance metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)


# Print performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predictions)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color="blue", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Plot Distribution of Predictions
real_preds = predictions[true_labels == 1]
fake_preds = predictions[true_labels == 0]
plt.figure(figsize=(8, 6))
plt.hist(real_preds, bins=25, alpha=0.7, label="Real")
plt.hist(fake_preds, bins=25, alpha=0.7, label="Fake")
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.title("Distribution of Predictions")
plt.legend(loc="upper right")
plt.show()

# Plot Metrics as a Bar Chart
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Performance Metrics")
plt.show()


