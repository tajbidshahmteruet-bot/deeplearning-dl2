import torch
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, logvar):
	x = x.view(-1, 784)
	recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	return recon_loss + kl_loss


if __name__ == "__main__":
    from src.models.vae import VAE

    model = VAE()
    dummy = torch.rand(128, 1, 28, 28)
    x_recon, mu, logvar = model(dummy)
    loss = vae_loss(x_recon, dummy, mu, logvar)
    print("Total loss:", loss.item())
    print("Loss per image:", loss.item() / 128)