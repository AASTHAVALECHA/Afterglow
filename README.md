# AFTERGLOW: The Hidden Paintings Inside Noise



**AFTERGLOW** is a computational art project that reveals the hidden abstract paintings inside photographs by capturing the liminal states of forward diffusion — the moment where representational content has vanished, but the color temperature and emotional weight of the original still linger.



## The Concept

Diffusion models learn to generate images by first learning to destroy them. During training, an image is progressively corrupted with Gaussian noise until nothing remains but static. Everyone focuses on the reverse process — generation. **AFTERGLOW focuses on the forward process — destruction.**

At ~40–60% dissolution, something extraordinary happens: all structure is gone, but **color is the last thing to survive entropy**. A sunset's amber persists long after its horizon line disappears. A portrait's warmth outlives the face it described.

These "afterglow" states are accidentally beautiful abstract paintings — works no human composed and no AI was prompted to create.

## How It Works

```
Source Image → VAE Encode → Add Noise (DDPM Schedule) → VAE Decode → Afterglow
```

The pipeline is deliberately simple:

1. **Encode** a source image into latent space via a diffusion model's VAE
2. **Inject Gaussian noise** at precisely calibrated timesteps using the DDPM scheduler
3. **Decode** back to pixel space

No denoising. No text prompts. No generation. The diffusion model is used purely for its noise schedule and its VAE encoder/decoder.

Additionally, dominant color palettes are extracted at each dissolution stage via k-means clustering, visualizing how chromatic identity persists while structure collapses.

## Installation

```bash
pip install torch diffusers transformers accelerate Pillow numpy scikit-learn
```

## Usage

### Generate afterglows from a single image
```bash
python afterglow.py --image sunset.jpg --output results/
```

### Generate with custom timesteps
```bash
python afterglow.py --image painting.jpg --output results/ --timesteps 200 400 600 800
```

### Generate gallery layouts and prints
```bash
python afterglow.py --image photo.jpg --output results/ --compose-strip --compose-gallery --compose-prints
```

### Demo mode (no input needed)
```bash
python afterglow.py --demo
```

## Output

For each source image, the pipeline produces:

- **Individual afterglows** at 9 dissolution stages (t=100 to t=980)
- **Dissolution strip** — horizontal sequence from source through afterglow to noise, with palette bars
- **Gallery grid** — curated selection of afterglow states
- **Standalone prints** — gallery-ready compositions with thin palette strip
- **Palette data** — extracted dominant colors at each stage

## Gallery

| Source | Afterglow (42% dissolution) |
|--------|---------------------------|
| Purple Twilight | Lavender-pink atmosphere, tree ghosts |
| City Rooftop Dusk | Blue-to-orange fire, cityscape shadow |
| Fuji & Cherry Blossoms | Pink wash, red pagoda ghost |
| Van Gogh — Starry Night | Deep blue field, moon glow persists |

## The Math

For a source image **x₀**, the afterglow at timestep **t** is:

```
A(x₀, t) = Decode( √(ᾱₜ) · Encode(x₀) + √(1 - ᾱₜ) · ε )
```

where **ᾱₜ** is the cumulative noise schedule coefficient and **ε** is sampled Gaussian noise.

The "sweet spot" for afterglows is typically **t ∈ [400, 800]** — the narrow band where form has dissolved but color still persists.

## Art Historical Context

AFTERGLOW connects to a lineage of dissolution in art:

- **Turner** — atmospheric dissolution of form into light
- **Monet** (late water lilies) — form dissolving into pure color impression
- **Rothko** — color fields as emotional experience stripped of representation
- **Richter** (squeegee paintings) — blurring photographs into chromatic abstraction
- **Mono no aware** (物の哀れ) — the Japanese aesthetic of beauty in impermanence


## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Diffusers
- GPU recommended (CPU works but slower)
- ~4GB VRAM for Stable Diffusion 2.1 VAE

## License

This project is released for artistic and research purposes. The source images used in the exhibition are either original photographs or public domain works.


