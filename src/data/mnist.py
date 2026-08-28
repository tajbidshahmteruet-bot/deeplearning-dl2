import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor() # Transfor to the tensor.

train_dataset = datasets.MNIST(root="data/raw", 				# Downloading training set
				train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root="data/raw", 					# Downloading test set
				train=False, download=True, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)



if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Pixel range:", images.min().item(), "to", images.max().item())
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader))