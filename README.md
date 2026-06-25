# Histogram-constrained Image Generation (HIG)

Project page for **Histogram-constrained Image Generation** (ECCV 2026).

> A training-free, plug-and-play control mechanism that enforces user-specified
> distributional constraints (e.g., color histograms or latent-token
> distributions) on diffusion models with **exact precision**, by modeling
> control as an optimal transport (OT) problem.

🔗 **Live page:** https://maps-research.github.io/hig/

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Structure

```
index.html            # the project page
static/css/           # bulma + custom styles
static/js/            # interactivity (copy BibTeX, scroll-to-top)
static/images/        # figures
```

Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
