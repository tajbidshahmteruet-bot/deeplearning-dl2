import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
	def __init__(self):
		super().__init__()
		self.latent_dim = 2

		# Encoder: 784 -> 400 -> latent_dim (mu, logvar)
		self.fc1 = nn.Linear(784, 400)
		self.fc_mu = nn.Linear(400, self.latent_dim)
		self.fc_logvar = nn.Linear(400, self.latent_dim)

		# Decoder latent_dim (mu, logvar) -> 400 -> 784
		self.fc3 = nn.Linear(self.latent_dim, 400)
		self.fc4 = nn.Linear(400, 784)

	def encoder(self, x):
		h = F.relu(self.fc1(x))
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)

		return mu, logvar

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z):
		h = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h))

	def forward(self, x):
		x = x.view(-1, 784)
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		x_recon = self.decode(z)

		return x_recon, mu, logvar


if __name__ == "__main__":
    model = VAE()
    dummy = torch.randn(128, 1, 28, 28)
    x_recon, mu, logvar = model(dummy)
    print("Input shape:     ", dummy.shape)
    print("Recon shape:     ", x_recon.shape)
    print("mu shape:        ", mu.shape)
    print("logvar shape:    ", logvar.shape)
    print("Recon value range:", x_recon.min().item(), "to", x_recon.max().item())
    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", n_params)