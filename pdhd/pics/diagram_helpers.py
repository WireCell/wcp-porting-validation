#!/usr/bin/env python3
"""Shared drawing helpers for the PDHD NF and SP algorithm diagrams.

16x9 "world" on a 19.2x10.8in figure => 1.2 in per world-unit on both axes,
so world units are physically square and placed inset images keep their aspect.
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

W, H = 16.0, 9.0
FIGIN = (19.2, 10.8)
INK = "#20242b"


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=FIGIN)
        self.fig.patch.set_facecolor("white")
        ov = self.fig.add_axes([0, 0, 1, 1])
        ov.set_xlim(0, W)
        ov.set_ylim(0, H)
        ov.axis("off")
        self.ov = ov

    # -- primitives --------------------------------------------------------
    def group_bg(self, x0, y0, w, h, ec, fc, label=None, lx=None, ly=None,
                 fs=13, alpha=0.9):
        self.ov.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.6, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=1))
        if label:
            self.ov.text(lx if lx is not None else x0 + w / 2,
                         ly if ly is not None else y0 + h - 0.28, label,
                         ha="center", va="center", fontsize=fs,
                         fontweight="bold", color=ec, zorder=2)

    def box(self, cx, cy, w, h, text, fc, ec, fs=12.5, tc=None, bold=True,
            lw=2.0):
        self.ov.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=5))
        self.ov.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color=tc or INK, zorder=6,
                     fontweight="bold" if bold else "normal")

    def algobox(self, cx, cy, w, h, title, bullets, fc, ec, tc=None,
                title_fs=12.5, bullet_fs=9.8, lw=2.0, dy=0.295):
        """Rounded box with a bold title band and bulleted sub-steps."""
        self.ov.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=5))
        self.ov.text(cx, cy + h / 2 - 0.27, title, ha="center", va="center",
                     fontsize=title_fs, fontweight="bold", color=tc or ec,
                     zorder=6)
        y = cy + h / 2 - 0.60
        for b in bullets:
            self.ov.text(cx - w / 2 + 0.22, y, b, ha="left", va="center",
                         fontsize=bullet_fs, color=INK, zorder=6)
            y -= dy

    def arrow(self, p0, p1, color, lw=2.6, ms=22, ls="-"):
        self.ov.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
            color=color, linestyle=ls, shrinkA=2, shrinkB=2, zorder=4))

    def leader(self, p_from, p_to, color):
        self.ov.add_patch(FancyArrowPatch(
            p_from, p_to, arrowstyle="-", linewidth=1.4, color=color,
            alpha=0.85, linestyle=(0, (5, 3)), zorder=3))
        self.ov.add_patch(Circle(p_to, 0.055, color=color, zorder=6))

    def title(self, main, sub):
        self.ov.text(W / 2, 8.62, main, ha="center", va="center",
                     fontsize=22, fontweight="bold", color=INK)
        self.ov.text(W / 2, 8.06, sub, ha="center", va="center",
                     fontsize=14.5, color="#444")

    def footer(self, text):
        self.ov.text(W / 2, 0.32, text, ha="center", va="center",
                     fontsize=9, color="#8a8a8a")

    # -- image inset -------------------------------------------------------
    def place_image(self, path, cx, wd, yb, caption, target, color, box=None):
        im = Image.open(path).convert("RGB")
        if box:
            w, h = im.size
            l, t, r, b = box
            im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
        iw, ih = im.size
        hd = wd * ih / iw
        ax = self.fig.add_axes([(cx - wd / 2) / W, yb / H, wd / W, hd / H])
        ax.imshow(np.asarray(im))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(1.8)
        self.ov.text(cx, yb - 0.20, caption, ha="center", va="top",
                     fontsize=11, color=color, fontweight="bold")
        if target is not None:
            self.leader((cx, yb + hd + 0.04), target, color)
        return hd

    def save(self, out_base):
        self.fig.savefig(out_base + ".png", dpi=200, facecolor="white")
        self.fig.savefig(out_base + ".pdf", facecolor="white")
        print("wrote", out_base + ".png / .pdf")
