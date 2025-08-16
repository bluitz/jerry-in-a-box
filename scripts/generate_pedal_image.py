import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI


DEFAULT_PROMPT = (
    "Photorealistic product photo of a custom guitar pedal sitting on top of a Raspberry Pi 5 board. "
    "The pedal has an aluminum enclosure, two footswitches, clear audio input and output 1/4-inch jacks, "
    "and a small blue OLED screen showing chord names. Clean desk background, soft studio lighting, "
    "shallow depth of field, no brand logos."
)


def _generate_with_openai(prompt: str, out_path: Path, size: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json",
    )

    b64_data = result.data[0].b64_json
    image_bytes = base64.b64decode(b64_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = out_path.with_suffix(".png")
    with open(tmp_png, "wb") as f:
        f.write(image_bytes)

    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        from PIL import Image
        with Image.open(tmp_png) as im:
            rgb_im = im.convert("RGB")
            rgb_im.save(out_path, format="JPEG", quality=92)
        try:
            tmp_png.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        if out_path.suffix.lower() == ".png":
            tmp_png.rename(out_path)


def _generate_placeholder(out_path: Path, size: str) -> None:
    # size format: WxH
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate a placeholder image. Install with `pip install Pillow`.") from exc

    width, height = map(int, size.lower().split("x"))
    img = Image.new("RGB", (width, height), color=(18, 18, 22))
    draw = ImageDraw.Draw(img)

    # Raspberry Pi board rectangle (bottom layer)
    board_margin = int(width * 0.08)
    board_rect = [board_margin, int(height*0.55), width - board_margin, height - board_margin]
    draw.rounded_rectangle(board_rect, radius=24, fill=(35, 85, 55))
    draw.text((board_margin+20, int(height*0.55)+20), "Raspberry Pi 5", fill=(220, 240, 230))

    # Pedal on top
    pedal_w, pedal_h = int(width*0.7), int(height*0.38)
    pedal_x = (width - pedal_w)//2
    pedal_y = int(height*0.12)
    pedal_rect = [pedal_x, pedal_y, pedal_x+pedal_w, pedal_y+pedal_h]
    draw.rounded_rectangle(pedal_rect, radius=28, fill=(200, 205, 210), outline=(140, 140, 145), width=4)

    # OLED screen
    oled_w, oled_h = int(pedal_w*0.35), int(pedal_h*0.22)
    oled_x = pedal_x + int(pedal_w*0.08)
    oled_y = pedal_y + int(pedal_h*0.18)
    draw.rounded_rectangle([oled_x, oled_y, oled_x+oled_w, oled_y+oled_h], radius=10, fill=(20, 45, 100))
    draw.text((oled_x+16, oled_y+12), "G  C  D", fill=(150, 200, 255))

    # Audio jacks
    jack_r = int(pedal_h*0.06)
    in_center = (pedal_x + pedal_w - int(pedal_w*0.18), pedal_y + int(pedal_h*0.32))
    out_center = (pedal_x + pedal_w - int(pedal_w*0.18), pedal_y + int(pedal_h*0.62))
    for cx, cy, label in [(in_center[0], in_center[1], "IN"), (out_center[0], out_center[1], "OUT")]:
        draw.ellipse([cx-jack_r, cy-jack_r, cx+jack_r, cy+jack_r], fill=(80, 80, 85), outline=(30, 30, 32))
        draw.text((cx+jack_r+10, cy-10), label, fill=(30, 30, 32))

    # Footswitches
    fs_r = int(pedal_h*0.08)
    fs1 = (pedal_x + int(pedal_w*0.32), pedal_y + int(pedal_h*0.78))
    fs2 = (pedal_x + int(pedal_w*0.52), pedal_y + int(pedal_h*0.78))
    for cx, cy in [fs1, fs2]:
        draw.ellipse([cx-fs_r, cy-fs_r, cx+fs_r, cy+fs_r], fill=(170, 170, 175), outline=(120, 120, 125))

    # Title
    title = "Jerry in a Box — Concept"
    draw.text((int(width*0.05), int(height*0.03)), title, fill=(235, 235, 240))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)


def generate_image(prompt: str, out_path: Path, size: str = "1024x1024") -> None:
    try:
        _generate_with_openai(prompt, out_path, size)
    except Exception as e:
        # Fallback to placeholder if OpenAI call fails or env var is missing
        print(f"OpenAI image generation failed ({e}). Creating a placeholder concept image instead.")
        # Ensure JPEG extension by default
        if out_path.suffix.lower() not in {'.jpg', '.jpeg'}:
            out_path = out_path.with_suffix('.jpg')
        _generate_placeholder(out_path, size)


def main():
    parser = argparse.ArgumentParser(description="Generate a pedal concept image using OpenAI Images API")
    parser.add_argument("--out", type=str, default="docs/images/pedal-concept.jpg", help="Output image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text for image generation")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image size, e.g., 1024x1024")
    args = parser.parse_args()

    out_path = Path(args.out)
    generate_image(args.prompt, out_path, size=args.size)
    print(f"Wrote image to {out_path}")


if __name__ == "__main__":
    main()


