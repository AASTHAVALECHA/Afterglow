"""
AFTERGLOW: The Hidden Paintings Inside Noise
=============================================

CVPR AI Art 2025/2026 Submission
Implementation Pipeline

Concept:
    Every image, when dissolved through forward diffusion, passes through
    a liminal state — no longer recognizable, yet still carrying the color
    temperature, compositional energy, and emotional weight of the original.
    
    These "afterglow" states are accidentally beautiful abstract paintings,
    born from the dissolution of masterworks. Form dies. Color persists.
    
Pipeline:
    1. Load a source image (masterwork painting, photograph, etc.)
    2. Encode into latent space via VAE
    3. Apply forward diffusion noise at carefully selected timesteps
    4. Decode back to pixel space — this is the "afterglow"
    5. Extract dominant color palettes at each stage
    6. Compose final presentation layouts

Requirements:
    pip install torch diffusers transformers accelerate Pillow numpy scikit-learn matplotlib

Usage:
    python afterglow.py --image "path/to/image.jpg" --output "output_dir"
    python afterglow.py --image "path/to/image.jpg" --output "output_dir" --timesteps 200 400 600 800
    python afterglow.py --image "path/to/image.jpg" --output "output_dir" --compose-strip
    python afterglow.py --demo  # Uses built-in color gradients as source
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"
IMAGE_SIZE = 512
DEFAULT_TIMESTEPS = [100, 200, 350, 500, 650, 800, 900, 950, 980]
PALETTE_COLORS = 6  # dominant colors to extract per stage
SWATCH_HEIGHT = 60  # height of color swatches in composite
STRIP_GAP = 4       # gap between frames in strip


def load_and_preprocess(image_path: str, size: int = IMAGE_SIZE) -> Image.Image:
    """Load image and resize to square for latent encoding."""
    img = Image.open(image_path).convert("RGB")
    # Center crop to square
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def extract_palette(image: Image.Image, n_colors: int = PALETTE_COLORS) -> list:
    """Extract dominant colors using k-means clustering."""
    from sklearn.cluster import KMeans
    
    arr = np.array(image.resize((64, 64))).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    km.fit(arr)
    
    # Sort by cluster size (most dominant first)
    labels, counts = np.unique(km.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = km.cluster_centers_[sorted_idx].astype(int)
    proportions = counts[sorted_idx] / counts.sum()
    
    return [(tuple(c), float(p)) for c, p in zip(colors, proportions)]


def create_palette_swatch(
    palette: list, width: int, height: int = SWATCH_HEIGHT
) -> Image.Image:
    """Create a proportional color swatch bar from palette."""
    swatch = Image.new("RGB", (width, height), (245, 240, 232))
    draw = ImageDraw.Draw(swatch)
    
    x = 0
    for color, proportion in palette:
        w = max(1, int(proportion * width))
        draw.rectangle([x, 0, x + w, height], fill=color)
        x += w
    
    # Fill any remaining pixels with last color
    if x < width and palette:
        draw.rectangle([x, 0, width, height], fill=palette[-1][0])
    
    return swatch


class AfterglowPipeline:
    """
    The core pipeline: forward diffusion + decode = afterglow states.
    """
    
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = None):
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[afterglow] Loading VAE from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae = self.vae.to(self.device).eval()
        
        print(f"[afterglow] Loading scheduler...")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Total timesteps in the schedule
        self.num_timesteps = self.scheduler.config.num_train_timesteps
        print(f"[afterglow] Ready. Scheduler has {self.num_timesteps} timesteps.")
    
    def encode(self, image: Image.Image) -> "torch.Tensor":
        """Encode PIL image to latent space."""
        import torch
        
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(tensor).latent_dist
            latent = latent_dist.sample() * self.vae.config.scaling_factor
        
        return latent
    
    def decode(self, latent: "torch.Tensor") -> Image.Image:
        """Decode latent back to PIL image."""
        import torch
        
        with torch.no_grad():
            decoded = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        
        arr = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def forward_diffuse(self, latent: "torch.Tensor", timestep: int) -> "torch.Tensor":
        """Apply forward diffusion: add noise at given timestep."""
        import torch
        
        noise = torch.randn_like(latent)
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)
        return noisy_latent
    
    def generate_afterglow(
        self, image: Image.Image, timesteps: list = None
    ) -> list:
        """
        Generate afterglow states for an image at various dissolution levels.
        
        Returns list of (timestep, afterglow_image, palette) tuples.
        """
        if timesteps is None:
            timesteps = DEFAULT_TIMESTEPS
        
        print(f"[afterglow] Encoding source image...")
        latent = self.encode(image)
        
        results = []
        for t in sorted(timesteps):
            print(f"[afterglow] Dissolving at t={t}/{self.num_timesteps}...")
            noisy = self.forward_diffuse(latent, t)
            afterglow = self.decode(noisy)
            palette = extract_palette(afterglow)
            results.append((t, afterglow, palette))
            print(f"  → Dominant color: RGB{palette[0][0]}")
        
        return results


def compose_dissolution_strip(
    source: Image.Image,
    results: list,
    title: str = "AFTERGLOW",
    show_palette: bool = True,
) -> Image.Image:
    """
    Compose a beautiful horizontal strip showing the dissolution sequence.
    
    Layout:
        [Source] → [t=100] → [t=200] → ... → [t=980]
        [palette] [palette] [palette]     [palette]
    """
    n = len(results) + 1  # +1 for source
    img_size = IMAGE_SIZE
    gap = STRIP_GAP
    
    palette_h = SWATCH_HEIGHT if show_palette else 0
    label_h = 40
    title_h = 80
    
    total_w = n * img_size + (n - 1) * gap
    total_h = title_h + img_size + gap + palette_h + label_h
    
    # Background: warm paper tone
    bg_color = (245, 240, 232)
    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a nice font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 28)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Light.ttf", 11)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Title
    draw.text((20, 20), title, fill=(26, 23, 20), font=title_font)
    
    y_img = title_h
    y_pal = y_img + img_size + gap
    y_label = y_pal + palette_h + 4
    
    # Source image
    source_resized = source.resize((img_size, img_size), Image.LANCZOS)
    canvas.paste(source_resized, (0, y_img))
    source_palette = extract_palette(source)
    if show_palette:
        swatch = create_palette_swatch(source_palette, img_size)
        canvas.paste(swatch, (0, y_pal))
    draw.text((img_size // 2 - 30, y_label), "SOURCE", fill=(138, 130, 121), font=label_font)
    
    # Afterglow stages
    for i, (t, ag_img, palette) in enumerate(results):
        x = (i + 1) * (img_size + gap)
        ag_resized = ag_img.resize((img_size, img_size), Image.LANCZOS)
        canvas.paste(ag_resized, (x, y_img))
        
        if show_palette:
            swatch = create_palette_swatch(palette, img_size)
            canvas.paste(swatch, (x, y_pal))
        
        # Dissolution percentage
        pct = int(t / 1000 * 100)
        label = f"t={t} ({pct}%)"
        draw.text((x + img_size // 2 - 40, y_label), label, fill=(138, 130, 121), font=small_font)
    
    return canvas


def compose_gallery_grid(
    source: Image.Image,
    results: list,
    title: str = "AFTERGLOW",
    highlight_timesteps: list = None,
) -> Image.Image:
    """
    Compose a gallery-style layout: source on left, selected afterglows as grid.
    For print / exhibition presentation.
    """
    if highlight_timesteps:
        results = [(t, img, p) for t, img, p in results if t in highlight_timesteps]
    
    # Pick 4 best stages for 2x2 grid
    if len(results) > 4:
        indices = np.linspace(0, len(results) - 1, 4, dtype=int)
        results = [results[i] for i in indices]
    
    size = IMAGE_SIZE
    margin = 40
    gap = 16
    
    # Layout: source (large) on left, 2x2 grid on right
    left_w = size + margin
    right_w = 2 * size + gap
    total_w = margin + left_w + right_w + margin
    total_h = margin + 80 + 2 * size + gap + margin
    
    bg = (245, 240, 232)
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 32)
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", 16)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Light.ttf", 12)
    except (IOError, OSError):
        title_font = label_font = sub_font = ImageFont.load_default()
    
    y_top = margin
    draw.text((margin, y_top), title, fill=(26, 23, 20), font=title_font)
    draw.text((margin, y_top + 44), "Form dissolves. Color persists.", fill=(138, 130, 121), font=sub_font)
    
    y_grid = y_top + 80
    
    # Source (left)
    src_display = source.resize((size, size), Image.LANCZOS)
    canvas.paste(src_display, (margin, y_grid))
    draw.text((margin, y_grid + size + 8), "Source", fill=(138, 130, 121), font=label_font)
    
    # Afterglow grid (right)
    x_start = margin + left_w
    for i, (t, ag_img, palette) in enumerate(results[:4]):
        row, col = divmod(i, 2)
        x = x_start + col * (size + gap)
        y = y_grid + row * (size + gap)
        ag_display = ag_img.resize((size, size), Image.LANCZOS)
        canvas.paste(ag_display, (x, y))
        
        pct = int(t / 1000 * 100)
        draw.text(
            (x, y + size + 8),
            f"Afterglow — {pct}% dissolution",
            fill=(138, 130, 121), font=label_font
        )
    
    return canvas


def compose_single_afterglow(
    afterglow_img: Image.Image,
    palette: list,
    timestep: int,
    size: int = 1024,
) -> Image.Image:
    """
    Compose a single afterglow as a standalone art print.
    Minimal: image + thin palette strip at bottom.
    """
    margin = 60
    pal_h = 24
    gap = 20
    
    total_w = size + 2 * margin
    total_h = margin + size + gap + pal_h + margin
    
    bg = (245, 240, 232)
    canvas = Image.new("RGB", (total_w, total_h), bg)
    
    # Image
    display = afterglow_img.resize((size, size), Image.LANCZOS)
    canvas.paste(display, (margin, margin))
    
    # Palette strip
    swatch = create_palette_swatch(palette, size, pal_h)
    canvas.paste(swatch, (margin, margin + size + gap))
    
    return canvas


def create_demo_source(style: str = "warm") -> Image.Image:
    """Create a beautiful gradient source image for demo mode (no dependencies)."""
    size = IMAGE_SIZE
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    
    if style == "warm":
        # Warm sunset gradient
        for y in range(size):
            for x in range(size):
                r = int(200 + 55 * (y / size) * np.sin(x / size * np.pi))
                g = int(80 + 120 * (1 - y / size))
                b = int(60 + 80 * (x / size))
                arr[y, x] = [min(255, r), min(255, g), min(255, b)]
    elif style == "cool":
        # Deep ocean gradient
        for y in range(size):
            for x in range(size):
                r = int(20 + 60 * (x / size))
                g = int(40 + 100 * np.sin(y / size * np.pi))
                b = int(120 + 135 * (y / size))
                arr[y, x] = [min(255, r), min(255, g), min(255, b)]
    elif style == "earth":
        # Rich earth tones
        for y in range(size):
            for x in range(size):
                r = int(140 + 80 * np.sin((x + y) / size * np.pi))
                g = int(90 + 60 * np.cos(y / size * np.pi * 2))
                b = int(50 + 40 * (x / size))
                arr[y, x] = [min(255, r), min(255, g), min(255, b)]
    
    return Image.fromarray(arr)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AFTERGLOW — The Hidden Paintings Inside Noise"
    )
    parser.add_argument("--image", type=str, help="Path to source image")
    parser.add_argument("--output", type=str, default="afterglow_output",
                        help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Diffusion model ID")
    parser.add_argument("--timesteps", type=int, nargs="+", default=DEFAULT_TIMESTEPS,
                        help="Timesteps for dissolution")
    parser.add_argument("--compose-strip", action="store_true",
                        help="Compose a horizontal dissolution strip")
    parser.add_argument("--compose-gallery", action="store_true",
                        help="Compose a gallery grid layout")
    parser.add_argument("--compose-prints", action="store_true",
                        help="Save each afterglow as standalone print")
    parser.add_argument("--demo", action="store_true",
                        help="Run with demo gradient images (no input needed)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load source image
    if args.demo:
        print("[afterglow] Demo mode — generating gradient source images")
        sources = {
            "warm_gradient": create_demo_source("warm"),
            "cool_gradient": create_demo_source("cool"),
            "earth_gradient": create_demo_source("earth"),
        }
    elif args.image:
        name = Path(args.image).stem
        sources = {name: load_and_preprocess(args.image)}
    else:
        print("Error: Provide --image or --demo")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = AfterglowPipeline(model_id=args.model, device=args.device)
    
    for name, source in sources.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {name}")
        print(f"{'='*60}")
        
        # Save source
        source.save(os.path.join(args.output, f"{name}_source.png"))
        
        # Generate afterglows
        results = pipeline.generate_afterglow(source, args.timesteps)
        
        # Save individual afterglows
        for t, ag_img, palette in results:
            ag_img.save(os.path.join(args.output, f"{name}_afterglow_t{t:04d}.png"))
            print(f"  Saved: {name}_afterglow_t{t:04d}.png")
        
        # Compose strip
        if args.compose_strip or True:  # Always make the strip
            strip = compose_dissolution_strip(source, results, title=f"AFTERGLOW — {name}")
            strip.save(os.path.join(args.output, f"{name}_strip.png"))
            print(f"  Saved: {name}_strip.png")
        
        # Compose gallery
        if args.compose_gallery:
            gallery = compose_gallery_grid(source, results, title="AFTERGLOW")
            gallery.save(os.path.join(args.output, f"{name}_gallery.png"))
            print(f"  Saved: {name}_gallery.png")
        
        # Compose individual prints
        if args.compose_prints:
            for t, ag_img, palette in results:
                print_img = compose_single_afterglow(ag_img, palette, t)
                print_img.save(os.path.join(args.output, f"{name}_print_t{t:04d}.png"))
            print(f"  Saved {len(results)} standalone prints")
        
        # Save palette data
        palette_data = []
        src_pal = extract_palette(source)
        palette_data.append({"stage": "source", "timestep": 0, "palette": src_pal})
        for t, _, palette in results:
            palette_data.append({"stage": f"t={t}", "timestep": t, "palette": palette})
        
        # Save as readable text
        with open(os.path.join(args.output, f"{name}_palettes.txt"), "w") as f:
            f.write(f"AFTERGLOW Palette Analysis: {name}\n")
            f.write("=" * 60 + "\n\n")
            for entry in palette_data:
                f.write(f"Stage: {entry['stage']}\n")
                for color, prop in entry['palette']:
                    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                    f.write(f"  {hex_color}  ({prop:.1%})\n")
                f.write("\n")
        
        print(f"  Saved: {name}_palettes.txt")
    
    print(f"\n[afterglow] Complete. All outputs in: {args.output}/")
    print("[afterglow] For best results, try with famous paintings as source images.")
    print("[afterglow] Suggested: Starry Night, Girl with a Pearl Earring, The Great Wave")


if __name__ == "__main__":
    main()
