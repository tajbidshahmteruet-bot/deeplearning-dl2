import torch
from torch import optim

from src.models.vae import VAE
from src.models.loss import vae_loss
from src.data.mnist import train_loader


def train(epochs=20, lr=0.001):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Training on:", device)
	model = VAE().to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	model.train()
	for epoch in range(1, epochs + 1):
		total_loss = 0
		for images, _ in train_loader:
			images = images.to(device)

			optimizer.zero_grad()
			x_recon, mu, logvar = model(images)
			loss = vae_loss(x_recon, images, mu, logvar)
			loss.backward()
			optimizer.step()


			total_loss += loss.item()

		avg = total_loss / len(train_loader.dataset)
		print(f"Epoch {epoch:2d} | avg loss per image: {avg:.2f}")

	torch.save(model.state_dict(), "experiments/vae_mnist.pth")
	print("Model saved to experiments/vae_mnist.pth")


if __name__ == "__main__":
	train()