"""Simple script to generate sample equation images for testing."""
from PIL import Image, ImageDraw, ImageFont
import os

SAMPLES = [
    "2x + 3 = 7",
    "x^2 - 5x + 6 = 0",
    "(x+2)(x-3) = 0",
    "\u00bd x^2 - 3x + 2 = 0"
]

def make_image(text, path):
    img = Image.new('RGB', (800, 200), color=(255,255,255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 36)
    except Exception:
        font = ImageFont.load_default()
    d.text((20,60), text, fill=(0,0,0), font=font)
    img.save(path)

def generate(folder='samples'):
    os.makedirs(folder, exist_ok=True)
    for i, s in enumerate(SAMPLES, start=1):
        make_image(s, os.path.join(folder, f'eq_{i}.png'))
    print('Generated sample images in', folder)

if __name__ == '__main__':
    generate()
