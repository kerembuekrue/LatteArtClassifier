import torch
from src.models.cnn import CNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import numpy as np

# Load the state dictionary into a new model instance
model = CNN()
model.load_state_dict(torch.load('../output/model_state.pth'))

#####################################
# TESTING
#####################################
model.eval()
# load testing dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
testing_dataset = datasets.ImageFolder(root='../data/testing_data/', transform=transform)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

# Get the class names from the dataset
class_names = testing_dataset.classes

# 5. Perform inference on the test set
all_preds = []
all_labels = []

# Get a batch of images for visualization
images, labels = next(iter(test_loader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Plot the first three images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    # Denormalize the image
    image = images[i].permute(1, 2, 0).cpu().numpy()
    image = (image * 0.5) + 0.5  # Denormalize

    # Ensure image values are in valid range
    image = np.clip(image, 0, 1)

    # Get the actual and predicted labels
    actual = class_names[labels[i]]
    predicted = class_names[preds[i]]
    title = f"Actual: {actual}\nPredicted: {predicted}"

    # Plot image with label info
    axes[i].imshow(image)
    axes[i].set_title(title)
    axes[i].axis('off')

# plt.tight_layout()
plt.savefig('../output/prediction_examples.png')
plt.show()

# Complete the evaluation on the whole test set
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
