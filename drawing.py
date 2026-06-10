import cv2
import numpy as np
import math
import random
import time

from constants import (
    C_WHITE, C_DIMGRAY,
    FONT, FONT_BOLD,
)

def rr(img, pt1, pt2, color, thick, r=20, alpha=1.0):
    """
    Menggambar persegi panjang dengan sudut membulat (rounded rectangle).

    img   : frame OpenCV yang akan digambar
    pt1   : sudut kiri atas  (x, y)
    pt2   : sudut kanan bawah (x, y)
    color : warna BGR
    thick : ketebalan garis; pakai -1 untuk isi penuh
    r     : radius sudut
    alpha : 0.0–1.0, kalau < 1.0 hasilnya semi-transparan
    """
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        r = 1

    if alpha < 1.0:
        overlay = img.copy()
        _rr_draw(overlay, x1, y1, x2, y2, color, -1, r)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        _rr_draw(img, x1, y1, x2, y2, color, max(1, thick), r)
    else:
        _rr_draw(img, x1, y1, x2, y2, color, thick, r)


def _rr_draw(img, x1, y1, x2, y2, color, thick, r):
    """Helper internal untuk rr() -- tidak perlu dipanggil langsung."""
    if thick == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thick)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thick)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thick)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thick)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thick)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thick)
        cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, color, thick)
        cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, color, thick)


def draw_text(img, msg, pos, scale=0.6, color=C_WHITE, thick=1, font=FONT, anchor="tl"):
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    x, y = pos
    if anchor == "center":
        x -= tw // 2
        y += th // 2
    if anchor == "tc":
        x -= tw // 2
    cv2.putText(img, msg, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def draw_text_with_shadow(img, msg, pos, scale=0.6, color=C_WHITE, thick=1,
                          font=FONT, anchor="tc"):
    (tw, _th), _ = cv2.getTextSize(msg, font, scale, thick)
    x, y = pos
    if anchor == "tc":
        x -= tw // 2

    cv2.putText(img, msg, (int(x + 1), int(y + 1)),
                font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, msg, (int(x), int(y)),
                font, scale, color, thick, cv2.LINE_AA)


def glow_circle(img, center, r, color, layers=3):
    for i in range(layers, 0, -1):
        alpha_val = 0.12 * i / layers
        overlay = img.copy()
        cv2.circle(overlay, center, r + i * 5, color, 1)
        cv2.addWeighted(overlay, alpha_val, img, 1 - alpha_val, 0, img)
    cv2.circle(img, center, r, color, 2, cv2.LINE_AA)
    cv2.circle(img, center, max(1, r - 3), color, -1)


def corner_bracket(img, x1, y1, x2, y2, color, size=28, thick=2):
    pts = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
    for cx, cy, dx, dy in pts:
        cv2.line(img, (cx, cy), (cx + dx * size, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * size), color, thick, cv2.LINE_AA)


def draw_separator(img, y, w, color=C_DIMGRAY):
    cv2.line(img, (0, y), (w, y), color, 1)

def scanline(img, alpha=0.025):
    h, w = img.shape[:2]
    for y in range(0, h, 4):
        img[y:y+1, :] = (img[y:y+1, :] * (1 - alpha)).astype(np.uint8)


def vignette(img, strength=0.40):
    h, w = img.shape[:2]
    k  = cv2.getGaussianKernel(h, h * 0.6)
    k2 = cv2.getGaussianKernel(w, w * 0.6)
    mask = k @ k2.T
    mask = mask / mask.max()
    mask = (1 - strength) + strength * mask
    img[:] = np.clip(img * mask[:, :, None], 0, 255).astype(np.uint8)

def pulse_val(period=1.5, lo=0.5, hi=1.0):
    t = time.time() % period / period
    v = (math.sin(t * 2 * math.pi) + 1) / 2
    return lo + v * (hi - lo)

class Particle:
    def __init__(self, x, y, color):
        angle       = random.uniform(0, 2 * math.pi)
        speed       = random.uniform(2, 7)
        self.x      = float(x)
        self.y      = float(y)
        self.vx     = math.cos(angle) * speed   
        self.vy     = math.sin(angle) * speed   
        self.life   = 1.0                       
        self.decay  = random.uniform(0.03, 0.07)
        self.radius = random.randint(2, 4)
        self.color  = color

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += 0.3    # gravitasi ke bawah
        self.life -= self.decay
        self.vx   *= 0.95   # gesekan horizontal

    def draw(self, img):
        if self.life <= 0:
            return
        a = max(0.0, self.life)
        c = tuple(int(v * a) for v in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.radius, c, -1, cv2.LINE_AA)
