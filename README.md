# Deep Learning: Generative Models & Detection from Scratch

Clean PyTorch implementations of generative and detection models, each paired with the mathematical derivation behind it. Built alongside JKU Linz's Deep Learning 2 course — the goal is genuine implementation fluency, not just calling `.fit()`.

Each model links **theory → code**: a derivation in `docs/` sits next to its implementation in `src/`, so the math and the mechanics stay connected.

## Topics

| Topic | What it covers | Key derivations | Status |
|-------|----------------|-----------------|--------|
| **Variational Autoencoders (VAEs)** | Latent-variable generative modeling, amortized inference | ELBO, reparameterization trick, Gaussian KL | 📋 Planned |
| **Generative Adversarial Networks (GANs)** | Adversarial training, min–max games | Optimal discriminator → JS divergence, Wasserstein distance | 📋 Planned |
| **Flow-Based Models** | Exact likelihood via invertible maps | Change of variables, triangular Jacobians, coupling layers (RealNVP) | 📋 Planned |
| **Autoregressive Models** | Sequential factorization of the joint distribution | Chain rule of probability, masked/causal attention | 📋 Planned |
| **Diffusion / Score-Based Models** | Iterative denoising generative processes | Forward/reverse diffusion, variational bound on likelihood | 📋 Planned |
| **Object Detection** | Localization + classification | IoU, RoI pooling vs. RoIAlign, mAP | 📋 Planned |

## Repository Structure

```
deeplearning-dl2/
├── src/
│   ├── models/      # model implementations
│   ├── utils/       # training loops, metrics, plotting
│   └── data/        # dataset loaders
├── notebooks/       # exploration and result visualizations
├── docs/            # derivations linked to each model
├── experiments/     # configs and logged runs
└── tests/           # sanity checks (shapes, gradient flow)
```

## Getting Started

```bash
pip install -r requirements.txt
```

Each model is runnable as a module, e.g.:

```bash
python -m src.models.vae --epochs 20
```

## Motivation

This repository is the applied counterpart to my theoretical study of deep learning. Where the coursework covers the derivations, this is where those equations get implemented, trained on toy data, broken, fixed, and mapped back to the math. It is a companion to my [STM32H753 bare-metal](https://github.com/tajbidshahmteruet-bot) work — one side systems and embedded, this side machine learning depth.

## License

MIT
