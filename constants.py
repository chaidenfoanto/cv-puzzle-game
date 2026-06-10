import cv2

C_ACCENT  = (210, 210, 210)   # abu-abu terang  → border, highlight UI
C_WHITE   = (240, 240, 240)   # putih           → teks utama
C_GRAY    = (140, 140, 140)   # abu sedang      → teks sekunder
C_DIMGRAY = (75,  75,   75)   # abu gelap       → garis pemisah, hints
C_DARK    = (10,  10,   10)   # hitam           # (belum dipakai, disimpan untuk nanti)
C_BG_HUD  = (8,   8,    8)   # hitam pekat     → latar panel HUD
C_ORANGE  = (200, 200, 200)   # (nama lama, isinya abu) → warna timer > 60 detik
C_GREEN   = (100, 220, 100)

FONT      = cv2.FONT_HERSHEY_SIMPLEX   # font biasa
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX    # font tebal

CAP_WIDTH  = 1280   # lebar frame kamera (piksel)
CAP_HEIGHT = 720    # tinggi frame kamera (piksel)
CAP_FPS    = 60     # target frame per detik

PINCH_THRESHOLD         = 45   # jarak piksel maksimal supaya dianggap pinch
PINCH_RELEASE_DEBOUNCE  = 6    # berapa frame "tidak pinch" sebelum benar-benar lepas
METAL_HOLD_REQUIRED     = 0.6  # detik tahan metal gesture supaya reset terpicu
CAPTURE_HOLD_TIME       = 0.8  # detik tahan pinch untuk mulai puzzle

GRID_SIZE = 3   # ukuran grid puzzle, 3 berarti 3×3 = 9 keping
