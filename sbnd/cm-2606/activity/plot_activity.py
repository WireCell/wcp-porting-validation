#!/usr/bin/env python
# Plot CPU% and memory(GB) vs time from an activity_logger/top.sh log
# (2 cols: %CPU %MEM).  Standalone (mirrors activity_logger/cpu-plot.py but
# saves PNGs and auto-scales the time axis from the measured run).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "lar-mon.log"
RAM_GB = 125.0           # host total RAM (free -g)
# time axis: elapsed wall seconds / number of samples (from RUN_INFO)
ELAPSED_S = 1781058649.735480981 - 1781058560.455725677   # = 89.2798 s
TITLE = "wcls-img-clus-matching-xin (full patrec), 2025f-mc event 1"

data = np.loadtxt(LOG)
cpu = data[:, 0]
mem_gb = data[:, 1] * RAM_GB / 100.0
n = len(cpu)
xscale = ELAPSED_S / n
t = np.arange(n) * xscale

print(f"samples={n} xscale={xscale:.4f} s/sample  "
      f"CPU max={cpu.max():.0f}% mean={cpu.mean():.0f}%  "
      f"MEM max={mem_gb.max():.2f} GB")

# memory
plt.figure(figsize=(10, 5.5))
plt.plot(t, mem_gb, color="#1f77b4", lw=1.2)
plt.grid(alpha=0.3); plt.xlabel("time [sec]", fontsize=14); plt.ylabel("memory [GB]", fontsize=14)
plt.title(TITLE + f"\nmax {mem_gb.max():.2f} GB", fontsize=12)
plt.tight_layout(); plt.savefig("activity_mem.png", dpi=130); plt.close()
print("wrote activity_mem.png")

# cpu
plt.figure(figsize=(10, 5.5))
plt.plot(t, cpu, color="#d62728", lw=1.0)
plt.grid(alpha=0.3); plt.xlabel("time [sec]", fontsize=14); plt.ylabel("cpu [%]", fontsize=14)
plt.title(TITLE + f"\nmax {cpu.max():.0f}%  mean {cpu.mean():.0f}%", fontsize=12)
plt.tight_layout(); plt.savefig("activity_cpu.png", dpi=130); plt.close()
print("wrote activity_cpu.png")
