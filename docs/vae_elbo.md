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

## 4. The Reparameterization Trick

To evaluate the reconstruction term we must sample $z \sim q(z|x)$. But
**sampling is not differentiable** — if $z$ comes straight from a random
number generator, gradients cannot flow back through it to the encoder,
breaking end-to-end training.

The trick rewrites the sample as a deterministic function of the
parameters plus external, parameter-free noise:

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```
Now the randomness lives in $\epsilon$, whose distribution does **not**
depend on $\mu$ or $\sigma$. So $z$ is an ordinary arithmetic function of
$\mu$ and $\sigma$ (with $\epsilon$ a fixed sampled constant), and the
gradients $\partial z / \partial \mu = 1$ and
$\partial z / \partial \sigma = \epsilon$ flow cleanly into the encoder.

**Key idea:** express the sample as a *deterministic, differentiable
transform of a parameter-free noise source*. The single stochastic line
is `torch.randn_like(std)` — and it draws from a fixed $\mathcal{N}(0, I)$,
independent of any weight.

## 5. The KL Divergence (Closed Form)

For two Gaussians — the encoder $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ and
the prior $p(z) = \mathcal{N}(0, I)$ — the KL divergence has a closed form.
For a single latent dimension $j$:

$$D_{KL} = \frac{1}{2}\left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)$$

Summing over all $J$ latent dimensions and rearranging the sign gives the
form used in code:

$$D_{KL}(q \parallel p) = -\frac{1}{2}\sum_{j=1}^{J}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

Term-by-term, the code **is** the formula:

| Math | Code | Note |
|------|------|------|
| $-\tfrac{1}{2}$ | `-0.5 *` | prefactor |
| $\sum$ | `torch.sum` | over latent dims (and batch) |
| $\log\sigma_j^2$ | `logvar` | stored directly as log-variance |
| $\mu_j^2$ | `mu.pow(2)` | |
| $\sigma_j^2$ | `logvar.exp()` | $\exp(\log\sigma^2) = \sigma^2$ |

**Sanity check:** $D_{KL} \geq 0$ always (it is a divergence). If your
computed KL is ever negative, there is a sign error. This invariant is
worth remembering for exams too.

**Common trap:** the $\sigma^2$ term is the *variance*, i.e.
`logvar.exp()` — not a raw output and not a standard deviation. Confusing
$\sigma^2$ with $\sigma$ (or with $\mathbb{E}[zz^\top]$ in the
moment-matching form) is a frequent mistake.