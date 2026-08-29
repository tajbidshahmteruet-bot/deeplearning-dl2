# VAE: From ELBO to Python Implementation

This document links the mathematics of the Variational Autoencoder to its implementation in `src/models/vae.py` and `src/models/loss.py`.


## 1. The Goal

We want a generative model that maximizes the likelihood of the data:

$$\log p(x) = \log \int p(x|z)\,p(z)\,dz$$

This integral is **intractable** - we cannot marginalize over all possible latent codes 'z'. The VAE sidesteps this by introducing an approximate posterior 'q(z|x)' (the encoder) and optimizing a tractable *lower bound* on 'log p(x)' instead: the Evidence Lower Bound (ELBO). 