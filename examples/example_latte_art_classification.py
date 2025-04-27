import torch
from src.models.cnn import CNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

#####################################
# LOAD AND PREPARE DATA
#####################################
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize all images
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    # transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])  # Normalize
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# load training dataset
training_dataset = datasets.ImageFolder(root='../data/training_data/', transform=transform)
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)

# Get a batch of images from the train_loader
# images, labels = next(iter(train_loader))
# Plot some data
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i in range(4):
#     image = images[i].permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
#     image = (image * 0.5) + 0.5  # Denormalize the image
#     axes[i].imshow(image)
#     axes[i].axis('off')  # Turn off axis
#     axes[i].set_title(f'Label: {labels[i].item()}')  # Display the label
# plt.show()

#####################################
# CHOOSE MODEL
#####################################
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

#####################################
# TRAIN MODEL
#####################################
model.train()
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    batch_count = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        batch_count += 1

    # Calculate average loss for the epoch
    avg_loss = running_loss / batch_count
    print("Epoch {:3d} - Avg Loss: {:.5e}".format(epoch, avg_loss))

#####################################
# TESTING
#####################################
model.eval()
# load testing dataset
testing_dataset = datasets.ImageFolder(root='../data/testing_data/', transform=transform)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

# 5. Perform inference on the test set
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient computation to save memory
    for images, labels in test_loader:
        outputs = model(images)  # Get model predictions
        _, preds = torch.max(outputs, 1)  # Get the predicted class (index with highest probability)
        all_preds.extend(preds.cpu().numpy())  # Collect predictions
        all_labels.extend(labels.cpu().numpy())  # Collect true labels

# 6. Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
