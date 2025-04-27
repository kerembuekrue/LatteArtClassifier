import torch
from src.models.cnn import CNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#####################################
# LOAD AND PREPARE DATA
#####################################
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize all images
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# load training dataset
dataset = datasets.ImageFolder(root='../data/training_data/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get a batch of images from the train_loader
images, labels = next(iter(train_loader))

# Plot some data
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i in range(3):
    image = images[i].permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = (image * 0.5) + 0.5  # Denormalize the image
    axes[i].imshow(image)
    axes[i].axis('off')  # Turn off axis
    axes[i].set_title(f'Label: {labels[i].item()}')  # Display the label
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
epochs = 10
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("Epoch {:4d} - Loss {:.4f}".format(epoch, loss))
