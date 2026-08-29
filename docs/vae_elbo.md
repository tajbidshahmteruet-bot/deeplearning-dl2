# VAE: From ELBO to Python Implementation

This document links the mathematics of the Variational Autoencoder to its implementation in `src/models/vae.py` and `src/models/loss.py`.


## 1. The Goal

We want a generative model that maximizes the likelihood of the data:

$$\log p(x) = \log \int p(x \mid z) \\, p(z) \\, dz$$

This integral is **intractable** - we cannot marginalize over all possible latent codes 'z'. The VAE sidesteps this by introducing an approximate posterior 'q(z|x)' (the encoder) and optimizing a tractable *lower bound* on 'log p(x)' instead: the Evidence Lower Bound (ELBO). 

## 2. The Evidence Lower Bound

Starting from `log p(x)` and introducing `q(z|x)`, a few lines of algebra
(multiply by 1 = q/q, apply Jensen's inequality) yield the ELBO:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \parallel p(z))$$

This bound has two interpretable terms:

- **Reconstruction term** $\mathbb{E}_{q(z|x)}[\log p(x|z)]$ — how well the
  decoder reconstructs the input from a latent code drawn by the encoder.

- **KL regularizer** $D_{KL}(q(z|x) \parallel p(z))$ — how far the encoder's
  distribution strays from the prior $p(z) = \mathcal{N}(0, I)$.

We **maximize** the ELBO. Since optimizers minimize, the training loss is the
**negative ELBO**: reconstruction error **plus** KL divergence.


$$\mathcal{L} = \underbrace{\text{BCE}(x, \hat{x})}_{\text{reconstruction}} + \underbrace{D_{KL}(q(z|x) \parallel p(z))}_{\text{regularizer}}$$

Implemented in `src/models/loss.py` as `recon_loss + kl_loss`.