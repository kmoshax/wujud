"""The README banner: a walking person drawn in CSI hex, the Wi-Fi waves that see them, and the name in ASCII.

    cd research && .venv/bin/python tools/banner.py   # writes docs/banner.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "..", "docs", "banner.png")
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNLNerdFontMono-Regular.ttf"
WIDTH, HEIGHT = 1720, 560
CELL_W, CELL_H = 11, 20
INK, DIM, FAINT = (238, 238, 238), (120, 120, 120), (48, 48, 48)
# the name in figlet's "univers" lettering, which is dense in the same way as the body
WORDMARK = r"""
                               88                      88
                               ""                      88
                                                       88
8b      db      d8 88       88 88 88       88  ,adPPYb,88
`8b    d88b    d8' 88       88 88 88       88 a8"    `Y88
 `8b  d8'`8b  d8'  88       88 88 88       88 8b       88
  `8bd8'  `8bd8'   "8a,   ,a88 88 "8a,   ,a88 "8a,   ,d88
    YP      YP      `"YbbdP'Y8 88  `"YbbdP'Y8  `"8bbdP"Y8
                              ,88
                            888P"
""".strip("\n").split("\n")
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

    # Wi-Fi waves leaving the board on the right and crossing the person
    board = (930, 280)
    for k, radius in enumerate(range(80, 620, 90)):
        steps = int(radius * 1.6 / CELL_H) + 3
        for angle in np.linspace(np.pi * 0.66, np.pi * 1.34, steps):
            x, y = board[0] + radius * np.cos(angle), board[1] + radius * 0.6 * np.sin(angle)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                draw.text((x, y), "(", fill=DIM if k < 3 else (72, 72, 72), font=font, anchor="mm")

    cols, rows = 52, 27
    person = shade(walking_person(cols * 8, rows * 16))
    for r, line in enumerate(ascii_rows(person, cols, rows, rng)):
        for c, ch in enumerate(line):
            if ch != " ":
                draw.rectangle((60 + c * CELL_W, 0 + r * CELL_H, 60 + (c + 1) * CELL_W, 0 + (r + 1) * CELL_H), fill=(0, 0, 0))
                draw.text((60 + c * CELL_W, 0 + r * CELL_H), ch, fill=INK, font=font)

    draw.text(board, "[esp32]", fill=INK, font=font, anchor="mm")
    for r, line in enumerate(WORDMARK):
        draw.text((1060, 180 + r * CELL_H), line, fill=INK, font=font)
    image.save(OUT)
    print(os.path.relpath(OUT))


if __name__ == "__main__":
    main()
