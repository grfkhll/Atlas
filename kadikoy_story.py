import cv2, numpy as np, subprocess, sys, math, wave
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]
W, H, FPS, SR = 1080, 1920, 30, 48000
rng = np.random.default_rng(11)

cap = cv2.VideoCapture(f"{D}/src/k.mp4"); SRC = []
while True:
    ok, f = cap.read()
    if not ok: break
    SRC.append(f)

# clips in source frames [a, b); transition INTO clip i: (type, overlap frames)
CLIPS = [(0, 240), (240, 284), (299, 428), (478, 627), (647, 715)]
TRANS = [None, ("zoom", 16), ("dissolve", 8), ("tear", 22), ("dissolve", 8)]
# timeline layout
STARTS = []; t = 0
for i, (a, b) in enumerate(CLIPS):
    if i: t -= TRANS[i][1]
    STARTS.append(t); t += b - a
TOTAL = t

def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def ease_in(x): x = min(max(x, 0), 1); return x ** 3
def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3

def zoom(img, s, cx=0.5, cy=0.5):
    if abs(s - 1) < 1e-3: return img
    nw, nh = int(W * s), int(H * s)
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = int(min(max(cx * nw - W / 2, 0), nw - W)); y0 = int(min(max(cy * nh - H / 2, 0), nh - H))
    return big[y0:y0 + H, x0:x0 + W]

def radial_blur(img, s, amt, cy):
    if amt < 0.01: return zoom(img, s, 0.5, cy)
    acc = np.zeros((H, W, 3), np.float32); n = 6
    for k in range(n): acc += zoom(img, s * (1 + amt * k / n), 0.5, cy)
    return (acc / n).astype(np.uint8)

def zoom_through(a, b, p):
    """Soft zoom-through: push into a, blend while both are zoomed, ease out of b."""
    cy = 0.33  # zoom toward her face
    za = radial_blur(a, 1 + 1.2 * ease_in(min(p / 0.6, 1)), 0.12 * ease(p / 0.6), cy)
    zb = radial_blur(b, 1 + 1.2 * (1 - ease_out(max((p - 0.4) / 0.6, 0))), 0.12 * (1 - ease((p - 0.4) / 0.6)), cy)
    m = ease((p - 0.35) / 0.3)
    return cv2.addWeighted(za, 1 - m, zb, m, 0)

def jag_profile(n, seed, amp):
    r = np.random.default_rng(seed)
    walk = np.cumsum(r.normal(0, 1, n)); walk -= np.linspace(walk[0], walk[-1], n)
    walk = walk / (np.abs(walk).max() + 1e-6) * amp
    return (walk + r.normal(0, amp * 0.15, n) + amp * 0.5 * np.sin(np.arange(n) / 37.0)).astype(np.float32)
JAG1 = jag_profile(W, 1, 34); JAG2 = jag_profile(W, 2, 18)
PAPER = np.full((H, W, 3), (222, 234, 242), np.float32)
PAPER += cv2.GaussianBlur(rng.normal(0, 12, (H, W)).astype(np.float32), (0, 0), 1.2)[..., None]
PAPER = np.clip(PAPER, 0, 255).astype(np.uint8)
YCOL = np.arange(H, dtype=np.float32)[:, None]

def paper_tear(a, b, p):
    """Soft paper cut: the page tears from top to bottom and drifts away, revealing b."""
    p = ease(p)
    edge = -80 + p * (H + 200)
    e1 = edge + JAG1[None, :]; e2 = e1 + 22 + JAG2[None, :]
    M = cv2.getRotationMatrix2D((W / 2, H), -ease_in(p) * 4, 1.0); M[1, 2] += ease_in(p) * 180
    a2 = cv2.warpAffine(a, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    sh = 1 - np.clip(1 - (e1 - YCOL) / 90, 0, 1) * 0.45
    bb = (b * np.where(YCOL < e1, sh, 1)[..., None]).astype(np.uint8)
    return np.where((YCOL < e1)[..., None], bb, np.where((YCOL < e2)[..., None], PAPER, a2)).astype(np.uint8)

# ---------- finishing: warm grade, vignette, grain ----------
x = np.arange(256, dtype=np.float32) / 255
def curve(lift, gain, contrast=0.12):
    y = x + contrast * (x - 0.5) * (1 - np.abs(2 * x - 1))
    return np.clip((lift + y * (gain - lift)) * 255, 0, 255).astype(np.uint8)
LUT = np.dstack([curve(0.04, 0.95), curve(0.025, 0.985), curve(0.015, 1.0)]).reshape(256, 1, 3)  # BGR: warm highlights, lifted blacks
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.32 * np.clip(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2, 0, 1))[..., None].astype(np.float32)
GRAIN = [cv2.GaussianBlur(rng.normal(0, 5, (H, W)).astype(np.float32), (0, 0), 0.7)[..., None] for _ in range(6)]
def finish(img, g):
    img = cv2.LUT(img, LUT)
    return np.clip(img.astype(np.float32) * VIG + GRAIN[g % 6], 0, 255).astype(np.uint8)

# ---------- text ----------
def rgba_text(txt, size, font="/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", pin=False):
    f = ImageFont.truetype(font, size)
    l, t, r, b = f.getbbox(txt)
    pad = 70 if pin else 0
    im = Image.new("RGBA", (r - l + 40 + pad, b - t + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if pin:  # map pin icon
        cx, cy, rr = 34, 20 + (b - t) / 2 - 8, 18
        d.ellipse((cx - rr + 3, cy - rr + 3, cx + rr + 3, cy + rr + 3), fill=(0, 0, 0, 90))
        d.polygon([(cx - 14, cy + 8), (cx + 14, cy + 8), (cx, cy + 34)], fill=(255, 255, 255, 255))
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(255, 255, 255, 255))
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(225, 90, 60, 255))
    d.text((20 - l + pad + 3, 20 - t + 4), txt, font=f, fill=(0, 0, 0, 110))
    d.text((20 - l + pad, 20 - t), txt, font=f, fill=(255, 255, 255, 255))
    return np.array(im)[:, :, [2, 1, 0, 3]]

def overlay(img, rgba, x0, y0, alpha=1.0):
    if alpha <= 0: return img
    h, w = rgba.shape[:2]; x0, y0 = int(x0), int(y0)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb: return img
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    m = sub[:, :, 3:4].astype(np.float32) / 255 * alpha
    img[ya:yb, xa:xb] = (img[ya:yb, xa:xb] * (1 - m) + sub[:, :, :3] * m).astype(np.uint8)
    return img

LOC = rgba_text("Kadıköy, İstanbul", 54, pin=True)
SUB = rgba_text("PAZARTESİ · 17:40", 34, "/usr/share/fonts/opentype/inter/Inter-Medium.otf")
END = "Bir sonraki pazartesiye kadar…"

def warm_glow(img, k):
    yy2, xx2 = np.mgrid[0:H:8, 0:W:8].astype(np.float32)
    cx, cy = W * (0.85 - 0.25 * k), H * 0.25
    d = np.exp(-((xx2 - cx) ** 2 + (yy2 - cy) ** 2) / (2 * 480 ** 2))
    d = cv2.resize(d, (W, H))[..., None] * 0.45 * math.sin(math.pi * k)
    return np.clip(img + d * np.array([40, 120, 255], np.float32), 0, 255).astype(np.uint8)

def clip_frame(i, n):
    a, b = CLIPS[i]
    return SRC[min(a + n, b - 1)]

def frame(g):
    i = max(k for k in range(len(CLIPS)) if STARTS[k] <= g)
    n = g - STARTS[i]
    if i + 1 < len(CLIPS) and g >= STARTS[i + 1]:
        pass
    img = clip_frame(i, n)
    # transition into clip i?
    if i > 0 and n < TRANS[i][1]:
        L = TRANS[i][1]; p = (n + 0.5) / L
        A = clip_frame(i - 1, g - STARTS[i - 1]); B = img
        kind = TRANS[i][0]
        if kind == "zoom": img = zoom_through(A, B, p)
        elif kind == "tear": img = paper_tear(A, B, p)
        else: img = cv2.addWeighted(A, 1 - ease(p), B, ease(p), 0)
    img = finish(np.ascontiguousarray(img), g)
    t = g / FPS
    # fade in from black
    if g < 12: img = (img * ease(g / 12)).astype(np.uint8)
    # location lower-third
    if 0.6 < t < 4.2:
        k_in = ease_out((t - 0.6) / 0.6); k_out = 1 - ease((t - 3.6) / 0.6)
        al = min(k_in, k_out)
        img = overlay(img, LOC, 60 - (1 - k_in) * 80, H - 330, al)
        img = overlay(img, SUB, 130 - (1 - k_in) * 80, H - 250, al * 0.9)
    # rain begins: soft lightning flicker right after the zoom-through lands
    lt = g - (STARTS[1] + TRANS[1][1])
    for c, s in ((2, 0.22), (7, 0.14)):
        if 0 <= lt - c < 4:
            a = s * (1 - (lt - c) / 4)
            img = cv2.addWeighted(img, 1 - a, np.full_like(img, (255, 240, 225)), a, 0)
    # café: warm glow after paper cut
    ct = g - (STARTS[3] + TRANS[3][1] - 6)
    if 0 <= ct < 50: img = warm_glow(img, ct / 50)
    # ending typewriter + fade to black
    et = TOTAL - g
    if et <= 62:
        chars = min(len(END), (62 - et) // 2 + 1)
        tx = rgba_text(END[:chars], 50, "/usr/share/fonts/opentype/inter/Inter-Medium.otf")
        full_w = rgba_text(END, 50, "/usr/share/fonts/opentype/inter/Inter-Medium.otf").shape[1]
        img = overlay(img, tx, (W - full_w) / 2, H - 300)
    if et <= 20: img = (img * ease(et / 20)).astype(np.uint8)
    return img

# ---------- audio ----------
def load_orig():
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{D}/src/k.mp4", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, 2).astype(np.float32) / 32768

def edit_audio(orig):
    spf = SR // FPS
    out = np.zeros((TOTAL * spf + SR, 2), np.float32)
    for i, (a, b) in enumerate(CLIPS):
        seg = orig[a * spf:b * spf].copy()
        st = STARTS[i] * spf
        if i:
            L = TRANS[i][1] * spf; r = np.linspace(0, 1, L)[:, None]
            seg[:L] *= np.sin(r * np.pi / 2)
        if i + 1 < len(CLIPS):
            L = TRANS[i + 1][1] * spf; r = np.linspace(0, 1, L)[:, None]
            seg[-L:] *= np.cos(r * np.pi / 2)
        out[st:st + len(seg)] += seg
    return out[:TOTAL * spf]

def music(n):
    """Continuous lo-fi bed: Rhodes-like chords, soft bass, brushed drums, vinyl crackle."""
    r = np.random.default_rng(5)
    out = np.zeros(n, np.float32); bpm = 78; beat = 60 / bpm
    def add(sig, at):
        a = int(at * SR); b = min(n, a + len(sig))
        if a < n: out[a:b] += sig[:b - a]
    def note(f, dur, vel):
        tt = np.arange(int(dur * SR)) / SR
        mod = np.sin(2 * np.pi * f * 1.0 * tt) * 1.2 * np.exp(-tt * 3)
        s = np.sin(2 * np.pi * f * tt + mod) * np.exp(-tt * 1.6)
        s += 0.25 * np.sin(2 * np.pi * f * 2 * tt) * np.exp(-tt * 4)
        return (s * vel * np.minimum(1, tt * 300) * np.minimum(1, (dur - tt) * 8)).astype(np.float32)
    mid = lambda m: 440 * 2 ** ((m - 69) / 12)
    prog = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]  # Fmaj7 Em7 Dm7 Cmaj7
    bass = [41, 40, 38, 36]
    mel = [72, 69, 71, 67, 69, 64, 67, 64]
    bar = 4 * beat; nbars = int(n / SR / bar) + 1
    for k in range(nbars):
        t0 = k * bar; ch = prog[k % 4]
        for j, m in enumerate(ch): add(note(mid(m), bar * 1.05, 0.07), t0 + j * 0.018)
        add(note(mid(ch[1] + 12), beat * 1.5, 0.035), t0 + 2.5 * beat)
        tb = np.arange(int(bar * SR)) / SR
        add((np.sin(2 * np.pi * mid(bass[k % 4]) * tb) * np.exp(-tb * 0.9) * np.minimum(1, tb * 200) * 0.22).astype(np.float32), t0)
        if k % 2 == 1:
            add(note(mid(mel[(k // 2) % 8]), beat * 2, 0.045), t0 + beat)
            add(note(mid(mel[(k // 2 + 3) % 8]), beat * 2, 0.035), t0 + 2.75 * beat)
        if k >= 1:  # drums enter after the first bar
            for bt in range(4):
                tk = np.arange(int(0.3 * SR)) / SR
                if bt in (0, 2) or (bt == 3 and k % 2):
                    kick = np.sin(2 * np.pi * (48 * tk + 60 * (1 - np.exp(-tk * 25)) / 25)) * np.exp(-tk * 10) * 0.35
                    add(kick.astype(np.float32), t0 + bt * beat + (0.5 * beat if bt == 3 else 0))
                if bt in (1, 3):
                    tn = np.arange(int(0.25 * SR)) / SR
                    z = r.normal(0, 1, len(tn)); z = z - np.convolve(z, np.ones(3) / 3, "same")
                    add((z * np.exp(-tn * 14) * 0.06).astype(np.float32), t0 + bt * beat)
                for h in (0, 0.5):
                    th = np.arange(int(0.05 * SR)) / SR
                    z = r.normal(0, 1, len(th)); z = z - np.convolve(z, np.ones(4) / 4, "same")
                    add((z * np.exp(-th * 90) * (0.03 if h else 0.022)).astype(np.float32), t0 + (bt + h) * beat + (0.06 * beat if h else 0))
    # soft lowpass for warmth (one-pole)
    y = np.zeros_like(out); acc = 0.0; a = 0.35
    for i in range(0, n):
        acc += a * (out[i] - acc); y[i] = acc
    crackle = (r.random(n) > 0.9993) * r.normal(0, 0.12, n)
    y += np.convolve(crackle, np.exp(-np.arange(30) / 6), "same").astype(np.float32) * 0.5
    y += cv2.GaussianBlur(r.normal(0, 1, (1, n)).astype(np.float32), (0, 0), 3)[0] * 0.004
    fi = int(1.2 * SR); fo = int(1.6 * SR)
    y[:fi] *= np.linspace(0, 1, fi); y[-fo:] *= np.linspace(1, 0, fo)
    return y

def thunder(n):
    out = np.zeros(n, np.float32); r = np.random.default_rng(9)
    at = int((STARTS[1] + TRANS[1][1] + 2) / FPS * SR); L = int(2.5 * SR)
    z = r.normal(0, 1, L); acc = 0; y = np.zeros(L)
    for i in range(L): acc += 0.012 * (z[i] - acc); y[i] = acc
    env = np.minimum(1, np.arange(L) / (0.08 * SR)) * np.exp(-np.arange(L) / (0.9 * SR))
    seg = (y * env * 9).astype(np.float32)
    out[at:at + L] += seg[:max(0, min(L, n - at))]
    return out

def build_audio():
    orig = edit_audio(load_orig())
    n = len(orig)
    mus = music(n) + thunder(n)
    mix = orig * 0.55 + mus[:, None] * 1.0
    peak = np.abs(mix).max(); mix = np.tanh(mix / max(peak, 1e-6) * 1.2) * 0.89
    with wave.open(f"{D}/work/mix.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    only = [int(a) for a in sys.argv[2:]]
    if only:
        for g in only: cv2.imwrite(f"{D}/work/f{g:04d}.jpg", frame(g))
        print("TOTAL", TOTAL, STARTS); sys.exit()
    build_audio()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-i", f"{D}/work/mix.wav", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", f"{D}/kadikoy_reels.mp4"], stdin=subprocess.PIPE)
    for g in range(TOTAL):
        enc.stdin.write(frame(g).tobytes())
    enc.stdin.close(); enc.wait()
