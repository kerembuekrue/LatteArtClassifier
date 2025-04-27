import torch
from src.models.cnn import CNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt

#####################################
# LOAD AND PREPARE DATA
#####################################
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# load training dataset
full_dataset = datasets.ImageFolder(root='../data/training_data/', transform=transform)

# Define the split ratio: 80% for training and 20% for validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Randomly split the dataset
training_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)


# Create DataLoader for training and validation sets
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#####################################
# CHOOSE MODEL
#####################################
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

#####################################
# TRAIN MODEL
#####################################
model.train()
epochs = 1
# Track the loss values
train_losses = []
val_losses = []
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

    avg_train_loss = running_loss / batch_count
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count
    val_losses.append(avg_val_loss)

    # Print losses every 10 epochs
    # if epoch % 10 == 0:
    print(f"Epoch {epoch:3d} - Train Loss: {avg_train_loss:.5e} - Val Loss: {avg_val_loss:.5e}")

    # # Calculate average loss for the epoch
    # avg_loss = running_loss / batch_count
    # if epoch % 10 == 0:
    #     print("Epoch {:3d} - Avg Loss: {:.5e}".format(epoch, avg_loss))

# Or save just the model state dictionary (recommended approach)
torch.save(model.state_dict(), '../output/model_state.pth')


# Plotting the loss curves
fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
ax.plot(range(epochs), train_losses, label='Training Loss', linewidth=2.0)
ax.plot(range(epochs), val_losses, label='Validation Loss', linewidth=2.0)
ax.set_xlabel('Epochs', fontsize=15)
ax.set_ylabel('Loss', fontsize=15)
ax.legend()
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig('../output/loss_curves.png')
