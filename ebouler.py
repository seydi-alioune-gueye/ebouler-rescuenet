#!/usr/bin/env python3
"""
ebouler.py — Détecte les bâtiments sur une image drone (type RescueNet)
et simule leur effondrement (éboulement).

Usage :
    python ebouler.py image.jpg
    python ebouler.py image.jpg --sortie resultats --graine 7 --intensite 1.3
    python ebouler.py image.jpg --ajout 1650,1240,2050,1932   # ajoute un bâtiment raté (x0,y0,x1,y1 en pixels)
    python ebouler.py image.jpg --seulement 2,5               # n'effondre que certains bâtiments (numéros de detection.png)

Sorties (dans --sortie) :
    detection.png    contours et numéros des bâtiments détectés
    eboulement.png   image finale, bâtiments effondrés
    avant_apres.png  comparaison côte à côte
    animation.gif    effondrement animé

Dépendances : opencv-python, numpy, scikit-image, scipy, imageio
Méthode de détection (sans apprentissage) :
  1. superpixels SLIC ;
  2. pour chaque superpixel : saturation, luminosité, contraste local et
     cohérence d'orientation (tenseur de structure) — les toits en tôle ont des
     nervures parallèles très cohérentes, les toits en bardeaux sont gris et texturés ;
  3. la route (zone grise, lisse et étendue) est exclue ;
  4. filtrage par forme : surface, solidité, rectangularité.
"""
import argparse, math, os
import cv2
import numpy as np
import imageio.v2 as imageio
from scipy.spatial import cKDTree
from skimage.segmentation import slic

WORK_W = 1288  # largeur de travail pour la détection


# ----------------------------------------------------------------------------- détection
def detecter_batiments(img, ajouts=()):
    H0, W0 = img.shape[:2]
    s = WORK_W / W0
    W, H = WORK_W, int(round(H0 * s))
    sm = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-3)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 5) - cv2.GaussianBlur(g, (0, 0), 5) ** 2, 0))

    # route : gris clair, lisse, très étendue
    road = ((hsv[..., 1] < 30) & (hsv[..., 2] > 150) & (std < 10)).astype(np.uint8) * 255
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(road)
    rm = np.zeros_like(road)
    for i in range(1, n):
        if st[i, 4] > 0.01 * W * H:
            rm[lab == i] = 255
    rm = cv2.morphologyEx(rm, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))

    segs = slic(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB), n_segments=1500, compactness=12, start_label=0)
    ns = segs.max() + 1
    idx = segs.ravel()
    cnt = np.bincount(idx, minlength=ns)
    mean = lambda a: np.bincount(idx, a.ravel(), ns) / cnt
    fc, fs, fv, fr, fd = mean(coh), mean(hsv[..., 1]), mean(hsv[..., 2]), mean(rm / 255.), mean(std)

    metal = (fc > 0.52) & (fs < 65) & (fv > 70) & (fr < 0.3)                       # tôle nervurée
    dark = (fs < 50) & (fv > 75) & (fv < 150) & (fd > 12) & (fc > 0.3) & (fr < 0.3)  # bardeaux

    def clean(sel, close, openk):
        m = sel[segs].astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((openk, openk), np.uint8))

    m = cv2.bitwise_or(clean(metal, 15, 21), clean(dark, 45, 31))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masks = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < 0.0065 * W * H:
            continue
        sol = a / max(cv2.contourArea(cv2.convexHull(c)), 1)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        if sol < 0.72 or a / max(rw * rh, 1) < 0.5:
            continue
        c = cv2.approxPolyDP(c, 4, True)
        mk = np.zeros((H, W), np.uint8)
        cv2.drawContours(mk, [c], -1, 255, -1)
        masks.append(cv2.resize(mk, (W0, H0), interpolation=cv2.INTER_NEAREST))

    for (x0, y0, x1, y1) in ajouts:  # bâtiments ajoutés à la main
        mk = np.zeros((H0, W0), np.uint8)
        mk[max(0, y0):min(H0, y1), max(0, x0):min(W0, x1)] = 255
        masks.append(mk)

    # ordre de lecture : haut → bas, gauche → droite
    def key(mk):
        ys, xs = np.nonzero(mk)
        return (int(ys.mean() // (H0 / 3)), xs.mean())
    masks.sort(key=key)
    return masks


# ----------------------------------------------------------------------------- effondrement
def ease_out_bounce(t):
    if t < 1 / 2.75: return 7.5625 * t * t
    if t < 2 / 2.75: t -= 1.5 / 2.75; return 7.5625 * t * t + .75
    if t < 2.5 / 2.75: t -= 2.25 / 2.75; return 7.5625 * t * t + .9375
    t -= 2.625 / 2.75; return 7.5625 * t * t + .984375


class Effondrement:
    """Prépare les fragments d'un bâtiment, puis sait le rendre à n'importe quel instant t ∈ [0,1]."""

    def __init__(self, img, mask, rng, intensite=1.0):
        self.mask = mask
        ys, xs = np.nonzero(mask)
        self.cx, self.cy = xs.mean(), ys.mean()
        self.x0, self.y0, self.x1, self.y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        size = math.sqrt(len(xs))
        self.size = size
        self.delay = rng.uniform(0, 0.25)  # chaque bâtiment tombe à son rythme

        # sol de gravats sous l'emprise du bâtiment
        H, W = mask.shape
        noise = rng.normal(0, 1, (H // 4 + 1, W // 4 + 1)).astype(np.float32)
        noise = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.2), (W, H))[..., None]
        # gravats = matériaux du toit assombris + teinte terre + texture
        roof_col = img[mask > 0].reshape(-1, 3).astype(np.float32).mean(0)
        base = 0.55 * roof_col * 0.55 + 0.45 * np.array([58, 72, 88], np.float32)
        self.rubble = np.clip(base + noise * 34, 0, 255)
        self.rubble_alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), size * .02)[..., None]

        # fragmentation de Voronoï du toit
        n_frag = int(np.clip(len(xs) / 2500, 25, 90))
        pick = rng.choice(len(xs), n_frag, replace=False)
        seeds = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)
        _, lab = cKDTree(seeds).query(np.stack([xs, ys], 1))
        self.frags = []
        for k in range(n_frag):
            sel = lab == k
            if sel.sum() < 30:
                continue
            fx, fy = xs[sel], ys[sel]
            bx0, by0, bx1, by1 = fx.min(), fy.min(), fx.max() + 1, fy.max() + 1
            patch = img[by0:by1, bx0:bx1].copy()
            alpha = np.zeros((by1 - by0, bx1 - bx0), np.float32)
            alpha[fy - by0, fx - bx0] = 1
            fcx, fcy = fx.mean(), fy.mean()
            # direction : vers l'extérieur + un peu de hasard ; certains restent sur place (effondrement en crêpe)
            dx, dy = fcx - self.cx, fcy - self.cy
            nrm = math.hypot(dx, dy) + 1e-6
            stay = rng.random() < 0.65
            dist = (rng.uniform(0.01, 0.06) if stay else rng.uniform(0.06, 0.22)) * size * intensite
            ang = math.atan2(dy, dx) + rng.normal(0, 0.5)
            self.frags.append(dict(
                patch=patch, alpha=alpha, box=(bx0, by0), c=(fcx - bx0, fcy - by0),
                d=(math.cos(ang) * dist, math.sin(ang) * dist),
                rot=rng.normal(0, 28 if not stay else 8) * intensite,
                scale=rng.uniform(0.72, 0.92),
                shade=rng.uniform(0.5, 0.85),
                order=rng.random()))
        self.frags.sort(key=lambda f: f["order"])

        # petits débris projetés
        nd = int(n_frag * 3 * intensite)
        r = size * rng.uniform(0.3, 0.7, nd) * intensite
        a = rng.uniform(0, 2 * math.pi, nd)
        self.debris = dict(x=self.cx + np.cos(a) * r, y=self.cy + np.sin(a) * r,
                           sx=rng.integers(1, 3, nd) * max(2, int(size * .008)),
                           rot=rng.uniform(0, 180, nd),
                           col=np.array([img[int(np.clip(self.cy + rng.normal(0, size * .2), 0, H - 1)),
                                             int(np.clip(self.cx + rng.normal(0, size * .2), 0, W - 1))]
                                         for _ in range(nd)]) * rng.uniform(.5, .9, (nd, 1)))

        # poussière
        dn = rng.normal(0, 1, (H // 8 + 1, W // 8 + 1)).astype(np.float32)
        self.dust_noise = cv2.resize(cv2.GaussianBlur(dn, (0, 0), 2), (W, H))
        yy, xx = np.mgrid[0:H, 0:W]
        self.dist = np.hypot(xx - self.cx, yy - self.cy).astype(np.float32)

    def render(self, canvas, t_global):
        t = np.clip((t_global - self.delay) / (1 - self.delay), 0, 1)
        if t <= 0:
            return
        fall = ease_out_bounce(t)
        H, W = canvas.shape[:2]

        # 1) le bâtiment disparaît → sol de gravats
        a = self.rubble_alpha * min(1, t * 2.5)
        canvas[:] = canvas * (1 - a) + self.rubble * a

        # 2) fragments du toit qui glissent, pivotent, rapetissent (ils tombent)
        for f in self.frags:
            ph, pw = f["alpha"].shape
            sc = 1 - (1 - f["scale"]) * fall
            M = cv2.getRotationMatrix2D(f["c"], f["rot"] * fall, sc)
            ox = f["box"][0] + f["d"][0] * fall
            oy = f["box"][1] + f["d"][1] * fall
            pad = int(max(ph, pw) * 0.6 + abs(f["d"][0]) + abs(f["d"][1])) + 4
            M[:, 2] += pad
            out_sz = (pw + 2 * pad, ph + 2 * pad)
            wp = cv2.warpAffine(f["patch"], M, out_sz, flags=cv2.INTER_LINEAR)
            wa = cv2.warpAffine(f["alpha"], M, out_sz, flags=cv2.INTER_LINEAR)
            gx0, gy0 = int(round(ox)) - pad, int(round(oy)) - pad
            cx0, cy0 = max(gx0, 0), max(gy0, 0)
            cx1, cy1 = min(gx0 + out_sz[0], W), min(gy0 + out_sz[1], H)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sl = (slice(cy0 - gy0, cy1 - gy0), slice(cx0 - gx0, cx1 - gx0))
            al = wa[sl][..., None]
            # ombre portée
            sh = int(self.size * 0.015 * (1 - fall * .5)) + 2
            ys, xs = slice(min(cy0 + sh, H), min(cy1 + sh, H)), slice(min(cx0 + sh, W), min(cx1 + sh, W))
            hh, ww = ys.stop - ys.start, xs.stop - xs.start
            if hh > 0 and ww > 0:
                canvas[ys, xs] *= 1 - al[:hh, :ww] * 0.45
            shade = 1 - (1 - f["shade"]) * fall
            reg = canvas[cy0:cy1, cx0:cx1]
            canvas[cy0:cy1, cx0:cx1] = reg * (1 - al) + wp[sl].astype(np.float32) * shade * al

        # 3) débris projetés
        db = self.debris
        k = int(len(db["x"]) * min(1, t * 1.5))
        for i in range(k):
            x = self.cx + (db["x"][i] - self.cx) * fall
            y = self.cy + (db["y"][i] - self.cy) * fall
            box = cv2.boxPoints(((x, y), (int(db["sx"][i] * 2.5), int(db["sx"][i])), db["rot"][i]))
            cv2.fillConvexPoly(canvas, box.astype(np.int32), db["col"][i].tolist())

        # 4) nuage de poussière : monte puis retombe
        puff = math.sin(math.pi * min(1, t * 1.15)) * 0.75 + 0.12 * t
        radius = self.size * (0.5 + 0.6 * t)
        dust = np.exp(-(self.dist / radius) ** 2) * (0.6 + 0.4 * self.dust_noise) * puff
        dust = np.clip(dust, 0, 0.85)[..., None]
        canvas[:] = canvas * (1 - dust) + np.array([168, 186, 200], np.float32) * dust


# ----------------------------------------------------------------------------- programme
def main():
    ap = argparse.ArgumentParser(description="Détecte les bâtiments et les fait s'ébouler.")
    ap.add_argument("image")
    ap.add_argument("--sortie", default="resultats")
    ap.add_argument("--graine", type=int, default=42, help="change l'aléatoire de l'effondrement")
    ap.add_argument("--intensite", type=float, default=1.0, help="0.5 = léger, 1.5 = violent")
    ap.add_argument("--ajout", action="append", default=[], help="x0,y0,x1,y1 d'un bâtiment à ajouter")
    ap.add_argument("--seulement", default="", help="numéros des bâtiments à effondrer, ex. 1,3")
    ap.add_argument("--images-gif", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.sortie, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Impossible de lire {args.image}")
    ajouts = [tuple(int(v) for v in a.split(",")) for a in args.ajout]

    masks = detecter_batiments(img, ajouts)
    print(f"{len(masks)} bâtiment(s) détecté(s)")

    # image de détection
    det = img.copy()
    for i, m in enumerate(masks, 1):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ov = det.copy(); cv2.drawContours(ov, cs, -1, (40, 40, 230), -1)
        det = cv2.addWeighted(ov, 0.25, det, 0.75, 0)
        cv2.drawContours(det, cs, -1, (40, 40, 230), 6)
        ys, xs = np.nonzero(m)
        p = (int(xs.mean()) - 25, int(ys.mean()) + 25)
        cv2.putText(det, str(i), p, cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 10, cv2.LINE_AA)
        cv2.putText(det, str(i), p, cv2.FONT_HERSHEY_DUPLEX, 2.4, (40, 40, 230), 4, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.sortie, "detection.png"), det)

    choix = {int(v) for v in args.seulement.split(",") if v.strip()}
    if choix:
        masks = [m for i, m in enumerate(masks, 1) if i in choix]

    rng = np.random.default_rng(args.graine)
    # image finale (pleine résolution)
    effs = [Effondrement(img, m, rng, args.intensite) for m in masks]
    final = img.astype(np.float32)
    for e in effs:
        e.render(final, 1.0)
    final = np.clip(final, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.sortie, "eboulement.png"), final)

    # avant / après
    w = 1100; h = int(img.shape[0] * w / img.shape[1])
    a = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(final, (w, h), interpolation=cv2.INTER_AREA)
    ba = np.full((h + 80, 2 * w + 24, 3), 22, np.uint8)
    ba[80:, :w] = a; ba[80:, w + 24:] = b
    cv2.putText(ba, "Avant", (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ba, "Apres eboulement", (w + 44, 55), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.sortie, "avant_apres.png"), ba)

    # animation (résolution réduite)
    gw = 900; sc = gw / img.shape[1]
    small = cv2.resize(img, (gw, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(args.graine)
    sm_masks = [cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) for m in masks]
    effs = [Effondrement(small, m, rng, args.intensite) for m in sm_masks if m.any()]
    frames = []
    n = args.images_gif
    for k in range(n + 8):
        t = min(1, k / n)
        c = small.astype(np.float32)
        if k > 0:
            for e in effs:
                e.render(c, t)
        frames.append(cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    durations = [0.6] + [0.07] * (n - 1) + [0.12] * 7 + [1.5]
    imageio.mimsave(os.path.join(args.sortie, "animation.gif"), frames, duration=durations[:len(frames)], loop=0)
    print("Fichiers écrits dans", args.sortie)


if __name__ == "__main__":
    main()
