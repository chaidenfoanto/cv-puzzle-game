import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

from constants import (
    C_ACCENT, C_GREEN, C_WHITE, C_GRAY, C_DIMGRAY, C_BG_HUD, C_ORANGE,
    FONT, FONT_BOLD,
    CAP_WIDTH, CAP_HEIGHT, CAP_FPS,
    # gesture
    PINCH_THRESHOLD, PINCH_RELEASE_DEBOUNCE, METAL_HOLD_REQUIRED, CAPTURE_HOLD_TIME,
    # puzzle
    GRID_SIZE,
)
from drawing import (
    rr, draw_text, draw_text_with_shadow,
    glow_circle, corner_bracket, draw_separator,
    scanline, vignette, pulse_val,
    Particle,
)

class PuzzleGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)

        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mode      = "setup"   # "setup" | "puzzle" | "solved"
        self.G         = GRID_SIZE

        # Data keping puzzle
        self.tiles      = []   # gambar tiap keping
        self.tile_rects = []   # posisi & ukuran tiap keping [x, y, w, h]
        self.tile_order = []   # urutan keping saat ini (index acak)
        self.locked     = []   # apakah tiap keping sudah di posisi benar

        # Drag-and-drop
        self.selected = None     # index keping yang sedang dipegang
        self.last_cur = (0, 0)   # posisi kursor terakhir (untuk drop)

        # Skor & waktu
        self.start_time = 0.0
        self.elapsed    = 0.0
        self.moves      = 0
        self.is_solved  = False

        # Cooldown setelah reset (supaya tidak langsung reset lagi)
        self.reset_cd = 0

        # Efek visual
        self.particles : list[Particle] = []
        self._score_pop = []   # teks "+1" yang melayang saat tile terkunci
        self._flash     = 0.0  # efek kilat putih saat puzzle dimulai

        # --- State setup: posisi & ukuran frame crop ---
        self.crop_cx   = None
        self.crop_cy   = None
        self.crop_size = 320

        # Nilai ter-smooth (EMA) untuk animasi crop yang mulus
        self._s_cx   = None
        self._s_cy   = None
        self._s_size = None
        self._ALPHA  = 0.18   # koefisien EMA: makin kecil = makin halus tapi lambat

        # State resize dengan pinch (satu tangan)
        self._prev_pinch_dist  = None
        self._pinch_size_start = None
        self._pinch_dist_start = None

        # Hold pinch untuk capture
        self._pinch_hold_start = None

        # FPS counter
        self._fps    = 0.0
        self._t_prev = time.time()

        # Debounce pinch (agar tile tidak drop karena tracking sesaat hilang)
        self._pinch_release_counter = 0
        self._pinch_stable          = False

        # Hold timer untuk metal gesture (agar tidak reset tidak sengaja) 
        self._metal_hold_start = None

    def _dist(self, a, b):
        """Jarak Euclidean antara dua titik (x, y)."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ema(self, old, new, a):
        return a * new + (1 - a) * old if old is not None else float(new)

    def _fps_update(self):
        now = time.time()
        self._fps    = 0.9 * self._fps + 0.1 * (1 / (now - self._t_prev + 1e-9))
        self._t_prev = now

    def _spawn_particles(self, cx, cy, color, n=18):
        for _ in range(n):
            self.particles.append(Particle(cx, cy, color))

    def _update_particles(self, img):
        alive = []
        for p in self.particles:
            p.update()
            p.draw(img)
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def _handle_crop_input(self, lms_list, w, h, is_pinching_list):
        n = len(lms_list)
        if n == 0:
            self._prev_pinch_dist = None
            return

        if n == 1:
            lms  = lms_list[0]
            idxs = [0, 5, 9, 13, 17]   # titik pangkal jari (palm center)
            px   = np.mean([lms.landmark[i].x for i in idxs]) * w
            py   = np.mean([lms.landmark[i].y for i in idxs]) * h

            idx_tip = (lms.landmark[8].x * w, lms.landmark[8].y * h)
            thm_tip = (lms.landmark[4].x * w, lms.landmark[4].y * h)
            pd      = self._dist(idx_tip, thm_tip)

            if is_pinching_list[0]:
                # Sedang pinch → pakai jarak perubahan untuk resize
                if self._prev_pinch_dist is None:
                    self._prev_pinch_dist  = pd
                    self._pinch_size_start = self.crop_size
                    self._pinch_dist_start = pd
                else:
                    if self._pinch_dist_start and self._pinch_dist_start > 5:
                        ratio          = pd / self._pinch_dist_start
                        new_size       = self._pinch_size_start * ratio
                        self.crop_size = float(np.clip(new_size, 150, min(w, h) * 0.92))
                self.crop_cx = float(px)
                self.crop_cy = float(py)
            else:
                # Tidak pinch → hanya geser posisi
                self._prev_pinch_dist  = None
                self._pinch_size_start = None
                self._pinch_dist_start = None
                self.crop_cx = float(px)
                self.crop_cy = float(py)
        else:
            # 2 tangan → jarak antar telunjuk = ukuran frame
            lms0, lms1 = lms_list[0], lms_list[1]
            p0 = (lms0.landmark[8].x * w, lms0.landmark[8].y * h)
            p1 = (lms1.landmark[8].x * w, lms1.landmark[8].y * h)
            self.crop_cx   = (p0[0] + p1[0]) / 2
            self.crop_cy   = (p0[1] + p1[1]) / 2
            d              = self._dist(p0, p1)
            self.crop_size = float(np.clip(d * 1.05, 150, min(w, h) * 0.92))
            self._prev_pinch_dist = None

        # Terapkan EMA supaya pergerakan mulus
        self._s_cx   = self._ema(self._s_cx,   self.crop_cx,   self._ALPHA)
        self._s_cy   = self._ema(self._s_cy,   self.crop_cy,   self._ALPHA)
        self._s_size = self._ema(self._s_size, self.crop_size, self._ALPHA)

    def _get_roi(self, w, h):
        if self._s_cx is None:
            cx, cy, s = w / 2, h / 2, self.crop_size
        else:
            cx, cy, s = self._s_cx, self._s_cy, self._s_size
        s  = int(s)
        x1 = int(np.clip(cx - s // 2, 0, w - s))
        y1 = int(np.clip(cy - s // 2, 0, h - s))
        return x1, y1, x1 + s, y1 + s

    def create_puzzle(self, frame, roi):
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2].copy()
        hc, wc = crop.shape[:2]
        th, tw = hc // self.G, wc // self.G

        self.tiles      = []
        self.tile_rects = []
        self.locked     = []

        for i in range(self.G):
            for j in range(self.G):
                tile = crop[i*th:(i+1)*th, j*tw:(j+1)*tw].copy()
                self.tiles.append(tile)
                self.tile_rects.append([j*tw + x1, i*th + y1, tw, th])
                self.locked.append(False)

        self.tile_order = list(range(len(self.tiles)))
        while True:
            random.shuffle(self.tile_order)
            if not all(self.tile_order[i] == i for i in range(len(self.tile_order))):
                break

        self.start_time = time.time()
        self.elapsed    = 0.0
        self.moves      = 0
        self.is_solved  = False
        self.particles  = []
        self._flash     = 0.8   # kilat putih saat puzzle baru dimulai

    def check_lock(self):
        newly_locked = []
        for i in range(len(self.tile_order)):
            was = self.locked[i]
            if self.tile_order[i] == i:
                self.locked[i] = True
                if not was:
                    newly_locked.append(i)
        if all(self.locked):
            self.is_solved = True
            self.mode      = "solved"
        return newly_locked

    def is_metal(self, lms):
        index_up    = lms.landmark[8].y  < lms.landmark[6].y
        pinky_up    = lms.landmark[20].y < lms.landmark[18].y
        middle_down = lms.landmark[12].y > lms.landmark[10].y
        ring_down   = lms.landmark[16].y > lms.landmark[14].y
        return index_up and pinky_up and middle_down and ring_down

    # -----------------------------------------------------------------------
    # RENDERING: HUD setup
    # -----------------------------------------------------------------------

    def _draw_hud_setup(self, frame, w, h, is_pinching, any_hand, roi):
        x1, y1, x2, y2 = roi
        side = x2 - x1

        # Gelapkan area di luar kotak crop
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        dark = frame.copy()
        dark[:] = (dark * 0.28).astype(np.uint8)
        frame[mask == 0] = dark[mask == 0]

        # Border berkedip di sekitar kotak crop
        pv  = pulse_val(1.4, 0.5, 1.0)
        col = (int(C_ACCENT[0]*pv), int(C_ACCENT[1]*pv), int(C_ACCENT[2]*pv))
        corner_bracket(frame, x1, y1, x2, y2, col, size=32, thick=2)

        # Tanda silang di tengah kotak
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(frame, (cx-15, cy), (cx+15, cy), C_DIMGRAY, 1)
        cv2.line(frame, (cx, cy-15), (cx, cy+15), C_DIMGRAY, 1)

        # Label ukuran frame
        draw_text(frame, f"{side}px", (cx, y1-10), 0.45, C_GRAY, 1, anchor="tc")

        # Pesan instruksi sesuai kondisi tangan
        if not any_hand:
            msgs = [
                "SHOW YOUR HAND TO THE CAMERA",
                "1 hand: move frame    pinch: resize",
                "2 hands: spread to resize",
            ]
        elif is_pinching:
            msgs = [
                "PINCH to RESIZE FRAME",
                "Release to lock position",
                "Hold pinch to capture",
            ]
        else:
            msgs = [
                "Move your hand to reposition",
                "Pinch to resize the frame",
                "Hold pinch when ready to capture",
            ]

        # Panel instruksi semi-transparan di bawah layar
        panel_y = h - 90
        rr(frame, (16, panel_y), (w-16, h-10), (0, 0, 0), -1, r=10, alpha=0.92)
        draw_separator(frame, panel_y, w, C_DIMGRAY)

        for k, m in enumerate(msgs):
            sc = 0.62 if k == 0 else 0.50
            cl = C_WHITE if k == 0 else (190, 190, 190)
            tk = 2 if k == 0 else 1
            draw_text_with_shadow(frame, m, (w//2, panel_y + 18 + k*24),
                                  sc, cl, tk, anchor="tc")

        # Teks "HOLD TO CAPTURE" saat pinch aktif
        if is_pinching and any_hand:
            pv2 = pulse_val(0.5, 0.6, 1.0)
            c2  = (int(C_GREEN[0]*pv2), int(C_GREEN[1]*pv2), int(C_GREEN[2]*pv2))
            draw_text_with_shadow(frame, "HOLD TO CAPTURE", (w//2, y2 + 26), 0.65, c2, 2, FONT_BOLD, anchor="tc")

    def _draw_hud_puzzle(self, frame, w, h):
        hud_h = 52
        rr(frame, (0, 0), (w, hud_h), C_BG_HUD, -1, r=0, alpha=0.90)
        draw_separator(frame, hud_h, w, C_DIMGRAY)

        draw_text(frame, "PUZZLE", (20, 32), 0.60, C_ACCENT, 1, FONT_BOLD)

        # Timer (hijau < 60 detik, abu > 60 detik)
        t = int(self.elapsed)
        mm, ss = t // 60, t % 60
        tcol = C_GREEN if self.elapsed < 60 else C_ORANGE
        draw_text(frame, f"{mm:02d}:{ss:02d}", (w//2, 32), 0.60, tcol, 1, FONT_BOLD, anchor="tc")

        draw_text(frame, f"{self.moves} MOVES", (w-170, 32), 0.50, C_WHITE, 1)
        draw_text(frame, f"{self._fps:.0f}",    (w-30,  32), 0.42, C_DIMGRAY, 1, anchor="tc")

        # Progress bar (tile terkunci / total)
        locked_n = sum(self.locked)
        total    = len(self.locked)
        bx, by   = 20, hud_h + 8
        bw, bh   = w - 40, 4
        rr(frame, (bx, by), (bx+bw, by+bh), (30, 30, 30), -1, r=2)
        fill = int(bw * locked_n / total) if total else 0
        if fill > 0:
            rr(frame, (bx, by), (bx+fill, by+bh), C_GREEN, -1, r=2)
        draw_text(frame, f"{locked_n} / {total}", (bx, by-4), 0.36, C_GRAY, 1)

        # Hint kontrol di bawah layar
        hints = "pinch = grab    open hand = drop    metal = reset"
        draw_text(frame, hints, (w//2, h-10), 0.38, C_DIMGRAY, 1, anchor="tc")

    def _draw_tiles(self, frame, cursor):
        h, w = frame.shape[:2]

        # Gambar semua keping kecuali yang sedang dipegang
        for i in range(len(self.tile_order)):
            if i == self.selected:
                continue
            tx, ty, tw, th = self.tile_rects[i]

            if 0 <= ty < h - th and 0 <= tx < w - tw:
                frame[ty:ty+th, tx:tx+tw] = self.tiles[self.tile_order[i]]

            if self.locked[i]:
                # Border hijau + tint hijau tipis untuk tile yang sudah terkunci
                rr(frame, (tx, ty), (tx+tw, ty+th), C_GREEN, 1, r=4, alpha=0.0)
                overlay = frame.copy()
                cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), C_GREEN, -1)
                cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
            else:
                rr(frame, (tx, ty), (tx+tw, ty+th), C_DIMGRAY, 1, r=4)

        # Gambar keping yang sedang dipegang (di atas semua)
        if self.selected is not None and cursor:
            tx, ty, tw, th = self.tile_rects[self.selected]
            dx = max(0, min(cursor[0] - tw//2, w - tw))
            dy = max(0, min(cursor[1] - th//2, h - th))

            # Bayangan keping yang dipegang
            shadow = frame.copy()
            cv2.rectangle(shadow, (dx+6, dy+6), (dx+tw+6, dy+th+6), (0, 0, 0), -1)
            cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)

            frame[dy:dy+th, dx:dx+tw] = self.tiles[self.tile_order[self.selected]]
            rr(frame, (dx, dy), (dx+tw, dy+th), C_ACCENT, 2, r=6)

            # Highlight slot yang akan menjadi target swap
            for i, (rx, ry, rw, rh) in enumerate(self.tile_rects):
                if i == self.selected or self.locked[i]:
                    continue
                if rx < cursor[0] < rx+rw and ry < cursor[1] < ry+rh:
                    overlay2 = frame.copy()
                    cv2.rectangle(overlay2, (rx, ry), (rx+rw, ry+rh), C_ACCENT, -1)
                    cv2.addWeighted(overlay2, 0.14, frame, 0.86, 0, frame)
                    rr(frame, (rx, ry), (rx+rw, ry+rh), C_ACCENT, 1, r=4)

    def _draw_solved_overlay(self, frame, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (4, 4, 4), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        bw2, bh2 = 480, 240
        bx, by   = w//2 - bw2//2, h//2 - bh2//2
        rr(frame, (bx, by), (bx+bw2, by+bh2), C_BG_HUD, -1, r=16, alpha=0.96)
        rr(frame, (bx, by), (bx+bw2, by+bh2), C_DIMGRAY,  1, r=16)

        # Teks SOLVED berkedip berwarna hijau
        pv = pulse_val(1.2, 0.6, 1.0)
        cg = tuple(int(v * pv) for v in C_GREEN)
        draw_text(frame, "SOLVED", (w//2, by+58), 1.2, cg, 2, FONT_BOLD, anchor="tc")

        # Waktu dan jumlah langkah
        t = int(self.elapsed)
        mm, ss = t // 60, t % 60
        draw_text(frame, f"{mm:02d}:{ss:02d}", (w//2, by+105), 0.72, C_WHITE, 1, FONT_BOLD, anchor="tc")
        draw_text(frame, f"{self.moves} moves",  (w//2, by+138), 0.55, C_GRAY,  1, anchor="tc")

        # Peringkat berdasarkan waktu
        if   self.elapsed < 20: rank, rc = "PERFECT", C_WHITE
        elif self.elapsed < 40: rank, rc = "GREAT",   C_GREEN
        else:                   rank, rc = "NICE",    C_GRAY
        draw_text(frame, rank, (w//2, by+182), 0.80, rc, 2, FONT_BOLD, anchor="tc")

        draw_text(frame, "metal gesture to play again", (w//2, by+218), 0.42, C_DIMGRAY, 1, anchor="tc")

    def _draw_cursor(self, frame, cursor, is_pinching):
        if not cursor:
            return
        col = C_GREEN if is_pinching else C_ACCENT
        glow_circle(frame, cursor, 7 if is_pinching else 5, col, layers=2)

    def _draw_score_pop(self, frame):
        alive = []
        for txt, x, y, life in self._score_pop:
            if life > 0:
                a = min(life, 1.0)
                c = (int(C_GREEN[0]*a), int(C_GREEN[1]*a), int(C_GREEN[2]*a))
                draw_text(frame, txt, (x, int(y)), 0.55, c, 1, FONT_BOLD, anchor="tc")
                alive.append((txt, x, y - 1.5, life - 0.03))
        self._score_pop = alive

    def run(self):
        while True:
            ret, raw = self.cap.read()
            if not ret:
                break
            raw   = cv2.flip(raw, 1)  
            h, w  = raw.shape[:2]
            frame = raw.copy()

            self._fps_update()

            # ── Proses hand tracking ────────────────────────────────────────
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            lms_list      = []
            is_pinch_list = []
            cursor        = None
            metal_detected = False
            raw_pinch_seen = False

            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    lms_list.append(lms)

                    idx_tip = (lms.landmark[8].x * w, lms.landmark[8].y * h)
                    thm_tip = (lms.landmark[4].x * w, lms.landmark[4].y * h)
                    pd      = self._dist(idx_tip, thm_tip)
                    pinch   = pd < PINCH_THRESHOLD
                    is_pinch_list.append(pinch)

                    cur_pos       = (int(idx_tip[0]), int(idx_tip[1]))
                    self.last_cur = cur_pos

                    if pinch:
                        raw_pinch_seen = True
                        cursor         = cur_pos

                    if self.is_metal(lms):
                        metal_detected = True

            if raw_pinch_seen:
                self._pinch_release_counter = 0
                self._pinch_stable          = True
            else:
                self._pinch_release_counter += 1
                if self._pinch_release_counter >= PINCH_RELEASE_DEBOUNCE:
                    self._pinch_stable = False
            is_pinching = self._pinch_stable

            any_hand = len(lms_list) > 0

            if metal_detected:
                if self._metal_hold_start is None:
                    self._metal_hold_start = time.time()
                metal_held = time.time() - self._metal_hold_start
            else:
                self._metal_hold_start = None
                metal_held             = 0.0
            metal_detected = metal_detected and metal_held >= METAL_HOLD_REQUIRED

            # Reset game (metal gesture)
            if metal_detected and self.mode in ("puzzle", "solved") and self.reset_cd == 0:
                self.mode                   = "setup"
                self.is_solved              = False
                self._s_cx                  = self._s_cy = self._s_size = None
                self._prev_pinch_dist       = None
                self._pinch_hold_start      = None
                self._pinch_stable          = False
                self._pinch_release_counter = 0
                self._metal_hold_start      = None
                self.reset_cd               = 50

            # Mode: SETUP
            if self.mode == "setup":
                self._handle_crop_input(lms_list, w, h, is_pinch_list)
                roi          = self._get_roi(w, h)
                x1, y1, x2, y2 = roi

                self._draw_hud_setup(frame, w, h, is_pinching, any_hand, roi)

                # Gambar skeleton tangan
                if results.multi_hand_landmarks:
                    spec_j = self.mp_draw.DrawingSpec(color=C_ACCENT,    thickness=1, circle_radius=2)
                    spec_c = self.mp_draw.DrawingSpec(color=(45, 45, 45), thickness=1)
                    for lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS, spec_j, spec_c)

                self._draw_cursor(frame, cursor, is_pinching)

                # Progress lingkaran untuk "hold to capture"
                if is_pinching and any_hand:
                    if self._pinch_hold_start is None:
                        self._pinch_hold_start = time.time()
                    held = time.time() - self._pinch_hold_start
                    prog = min(held / CAPTURE_HOLD_TIME, 1.0)
                    cx2, cy2 = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.ellipse(frame, (cx2, cy2), (28, 28), -90, 0, int(360 * prog), C_GREEN, 2)
                    if prog >= 1.0:
                        self.create_puzzle(raw, roi)
                        self.mode              = "puzzle"
                        self._pinch_hold_start = None
                else:
                    self._pinch_hold_start = None

            # Mode: PUZZLE atau SOLVED
            elif self.mode in ("puzzle", "solved"):
                if not self.is_solved:
                    self.elapsed = time.time() - self.start_time

                self._draw_tiles(frame, cursor if not self.is_solved else None)
                self._draw_hud_puzzle(frame, w, h)
                self._update_particles(frame)
                self._draw_score_pop(frame)

                # Logika drag-and-drop tile
                if not self.is_solved:
                    if is_pinching and cursor:
                        # Ambil tile (kalau belum memegang apa-apa)
                        if self.selected is None:
                            for i, (rx, ry, rw, rh) in enumerate(self.tile_rects):
                                if self.locked[i]:
                                    continue
                                if rx < cursor[0] < rx+rw and ry < cursor[1] < ry+rh:
                                    self.selected = i
                                    break
                    else:
                        # Lepas tile → cek apakah di atas tile lain (swap)
                        if self.selected is not None:
                            lx, ly = self.last_cur
                            for i, (rx, ry, rw, rh) in enumerate(self.tile_rects):
                                if i == self.selected or self.locked[i]:
                                    continue
                                if rx < lx < rx+rw and ry < ly < ry+rh:
                                    # Tukar posisi dua tile
                                    self.tile_order[self.selected], self.tile_order[i] = \
                                        self.tile_order[i], self.tile_order[self.selected]
                                    self.moves += 1
                                    newly = self.check_lock()
                                    for ni in newly:
                                        rx2, ry2, rw2, rh2 = self.tile_rects[ni]
                                        self._spawn_particles(rx2 + rw2//2, ry2 + rh2//2, C_GREEN)
                                        self._score_pop.append(("+1", rx2 + rw2//2, ry2 + rh2//2 - 20, 1.2))
                                    break
                            self.selected = None

                # Gambar skeleton tangan (lebih redup saat puzzle)
                if results.multi_hand_landmarks and not self.is_solved:
                    spec_j = self.mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=2)
                    spec_c = self.mp_draw.DrawingSpec(color=(35, 35, 35), thickness=1)
                    for lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS, spec_j, spec_c)

                self._draw_cursor(frame, cursor, is_pinching)

                if self.is_solved:
                    self._draw_solved_overlay(frame, w, h)

            # Efek post-processing 
            if self._flash > 0:
                fl    = frame.copy()
                fl[:] = 255
                cv2.addWeighted(fl, self._flash, frame, 1 - self._flash, 0, frame)
                self._flash = max(0.0, self._flash - 0.07)

            vignette(frame, strength=0.32)
            scanline(frame, alpha=0.025)

            if self.reset_cd > 0:
                self.reset_cd -= 1

            # Tampilkan frame
            cv2.imshow("PUZZLE", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
