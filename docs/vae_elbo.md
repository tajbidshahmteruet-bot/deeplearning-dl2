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

## 3. The Encoder: q(z|x)

The encoder approximates the posterior $q(z|x)$ as a diagonal Gaussian.
It maps an input image to **two** vectors: a mean $\mu$ and a
log-variance $\log \sigma^2$, each of size `latent_dim`.

```python
def encode(self, x):
    h = F.relu(self.fc1(x))
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    return mu, logvar
```

A shared hidden layer (`fc1`) feeds two separate heads (`fc_mu`,
`fc_logvar`) — the network learns *where* the latent code sits ($\mu$)
and *how uncertain* it is ($\log \sigma^2$) from the same features.

**Why log-variance, not variance?** A variance must be positive, but a
linear layer can output any real number. By treating the output as
$\log \sigma^2 \in \mathbb{R}$, we recover a guaranteed-positive
$\sigma = \exp(\tfrac{1}{2}\log\sigma^2)$ later. This is a numerical
stability trick — it avoids ever forcing a raw layer output to stay
positive.