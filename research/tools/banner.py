"""The README banner: a walking person drawn in CSI hex, the Wi-Fi waves that see them, and the name.

    cd research && .venv/bin/python tools/banner.py   # writes docs/banner.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "..", "docs", "banner.png")
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNLNerdFontMono-Regular.ttf"
FONT_LIGHT = "/usr/share/fonts/TTF/JetBrainsMonoNLNerdFontMono-Light.ttf"
WIDTH, HEIGHT = 1600, 560
CELL_W, CELL_H = 11, 20
INK, DIM, FAINT = (238, 238, 238), (120, 120, 120), (48, 48, 48)
# light to dense; the dense end is the hex the board prints, so the body is made of CSI
RAMP = " .'`,:;-~=+*il1r7vcz4f2e6593a0b8d"


def capsule(px, py, a, b, r):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - ax - t * vx, py - ay - t * vy) - r


def walking_person(w, h):
    """Signed distance of a person mid-stride, side view, unit = person height."""
    y, x = np.mgrid[0:h, 0:w]
    px, py = 1.1 * x / h - 0.55 * w / h, 1.04 - 1.1 * y / h
    parts = [
        np.hypot(px - 0.03, py - 0.91) - 0.075,                                # head
        capsule(px, py, (0.0, 0.80), (0.015, 0.55), 0.095),                    # torso
        capsule(px, py, (0.0, 0.53), (0.13, 0.30), 0.062),                     # front thigh
        capsule(px, py, (0.13, 0.30), (0.17, 0.05), 0.05),                     # front shin
        capsule(px, py, (0.0, 0.53), (-0.08, 0.29), 0.062),                    # back thigh
        capsule(px, py, (-0.08, 0.29), (-0.20, 0.08), 0.042),                  # back shin
        capsule(px, py, (0.17, 0.04), (0.24, 0.035), 0.022),                   # front foot
        capsule(px, py, (-0.21, 0.07), (-0.15, 0.03), 0.022),                  # back foot
        capsule(px, py, (0.0, 0.78), (0.10, 0.62), 0.045),                     # front upper arm
        capsule(px, py, (0.10, 0.62), (0.16, 0.48), 0.04),                     # front forearm
        capsule(px, py, (0.0, 0.78), (-0.09, 0.63), 0.036),                    # back upper arm
        capsule(px, py, (-0.09, 0.63), (-0.13, 0.49), 0.04),                   # back forearm
    ]
    return np.min(parts, axis=0)


def shade(sd, light=(-0.6, 0.8)):
    """Distance inside the body read as height: a rounded, lit volume from a flat silhouette."""
    inside = np.clip(-sd / 0.05, 0, 1)
    gy, gx = np.gradient(-sd)
    norm = np.hypot(gx, gy) + 1e-9
    lit = np.clip((gx / norm * light[0] - gy / norm * light[1]) * (1 - inside) + inside, 0, 1)
    return np.where(sd < 0, 0.25 + 0.75 * lit * (0.4 + 0.6 * inside), 0.0)


def ascii_rows(image, cols, rows, rng):
    h, w = image.shape
    out = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            v = image[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols].mean()
            if v < 0.05:
                line += " "
                continue
            k = int(np.clip(v, 0, 0.999) * (len(RAMP) - 1))
            line += RAMP[max(1, min(len(RAMP) - 1, k + rng.integers(-2, 3)))]
        out.append(line)
    return out


def main():
    rng = np.random.default_rng(7)
    font = ImageFont.truetype(FONT, 17)
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # faint CSI rain behind everything: real-looking packet lines
    for row in range(HEIGHT // CELL_H):
        line = "".join(rng.choice(list("0123456789abcdef")) for _ in range(WIDTH // CELL_W))
        draw.text((0, row * CELL_H), line, fill=(14, 14, 14), font=font)

    # Wi-Fi waves leaving the board on the right and crossing the person
    board = (1010, 300)
    for k, radius in enumerate(range(80, 1000, 75)):
        steps = int(radius * 1.6 / CELL_H) + 3
        for angle in np.linspace(np.pi * 0.66, np.pi * 1.34, steps):
            x, y = board[0] + radius * np.cos(angle), board[1] + radius * 0.6 * np.sin(angle)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                draw.text((x, y), "(", fill=DIM if k < 4 else (72, 72, 72) if k < 8 else FAINT, font=font, anchor="mm")

    cols, rows = 52, 27
    person = shade(walking_person(cols * 8, rows * 16))
    for r, line in enumerate(ascii_rows(person, cols, rows, rng)):
        for c, ch in enumerate(line):
            if ch != " ":
                draw.rectangle((60 + c * CELL_W, 0 + r * CELL_H, 60 + (c + 1) * CELL_W, 0 + (r + 1) * CELL_H), fill=(0, 0, 0))
                draw.text((60 + c * CELL_W, 0 + r * CELL_H), ch, fill=INK, font=font)

    draw.text(board, "[esp32]", fill=INK, font=font, anchor="mm")
    big = ImageFont.truetype(FONT_LIGHT if os.path.exists(FONT_LIGHT) else FONT, 150)
    draw.text((1120, 230), "wujud", fill=INK, font=big, anchor="lm")
    small = ImageFont.truetype(FONT, 26)
    draw.text((1128, 330), "presence, without a camera", fill=DIM, font=small, anchor="lm")
    tiny = ImageFont.truetype(FONT, 15)
    draw.text((1128, 372), "wi-fi csi  ·  esp32  ·  physics-trained", fill=(90, 90, 90), font=tiny, anchor="lm")
    image.save(OUT)
    print(os.path.relpath(OUT))


if __name__ == "__main__":
    main()
