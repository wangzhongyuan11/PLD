"""Bounded display-only previews; never modifies inference inputs or labels."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

SIZE = (640, 480)


def thumbnail(value, nearest=False):
    im = value.copy() if isinstance(value, Image.Image) else Image.fromarray(value)
    im = im.convert('RGB')
    im.thumbnail(SIZE, getattr(Image, 'Resampling', Image).NEAREST if nearest else getattr(Image, 'Resampling', Image).BILINEAR)
    return im


def enhance(value):
    return ImageOps.autocontrast(thumbnail(value), cutoff=1)


def compose(images, titles=None, width=640, height=480):
    if len(images) != 4 or width < 640 or height < 480:
        raise ValueError('four panels and at least 640x480 required')
    titles = titles or ['Original', 'Display contrast only', 'Segmentation', 'Overlay / contour']
    canvas = Image.new('RGB', (width, height), (15, 28, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), 'PLD | FOUR VIEWS | Teaching demo - not for diagnosis', fill='white')
    top, bottom, gap = 24, 20, 6
    panel_w = (width - 3*gap)//2
    panel_h = (height-top-bottom-3*gap)//2
    for i, value in enumerate(images):
        x = gap+(i%2)*(panel_w+gap); y = top+gap+(i//2)*(panel_h+gap)
        draw.text((x+4, y+2), titles[i], fill=(140, 220, 220))
        im = thumbnail(value, nearest=i==2)
        im.thumbnail((panel_w, panel_h-22), getattr(Image, 'Resampling', Image).NEAREST if i==2 else getattr(Image, 'Resampling', Image).BILINEAR)
        canvas.paste(im, (x+(panel_w-im.width)//2, y+22+(panel_h-22-im.height)//2))
    draw.text((10, height-17), 'Contrast affects display only; original model input is retained.', fill=(180, 195, 205))
    return np.asarray(canvas).copy()


def skin_frame(folder):
    folder = Path(folder)
    def read(name):
        with Image.open(folder/(name+'.png')) as im:
            return thumbnail(im, nearest=name in ('mask', 'contour'))
    original = read('original') if (folder/'original.png').exists() else read('mask')
    titles = ['Original' if (folder/'original.png').exists() else 'Mask replay (no original)',
              'Display contrast only', 'Binary mask', 'Contour overlay']
    return compose([original, enhance(original), read('mask'), read('outlined')], titles)
