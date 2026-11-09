"""Plot measured retest data. Requires matplotlib; no hard-coded performance values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    args = parser.parse_args()
    root = args.results_directory
    data = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, mode in zip(axes, ('solver', 'pipeline')):
        rows = sorted((r for r in data if r['mode'] == mode), key=lambda r: r['grid'])
        x = np.arange(len(rows))
        for offset, side, color, title in [(-.19, 'A', '#546E7A', 'A: ordinary launches'),
                                            (.19, 'B', '#00897B', 'B: pressure graph')]:
            stats = [r['statistics']['completed_run_wall'][side] for r in rows]
            center = [s['median_mean_ms'] / r['frames'] for s, r in zip(stats, rows)]
            lo = [c - s['min_mean_ms'] / r['frames'] for c, s, r in zip(center, stats, rows)]
            hi = [s['max_mean_ms'] / r['frames'] - c for c, s, r in zip(center, stats, rows)]
            bars = ax.bar(x + offset, center, .36, label=title, color=color,
                          yerr=[lo, hi], capsize=4)
            ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
        ax.set_xticks(x, [f'{r["grid"]} x {r["grid"]}' for r in rows])
        ax.set_title(mode.capitalize())
        ax.set_ylabel('Completed workload wall time / step (ms)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.22)
    axes[0].legend(loc='upper left')
    fig.suptitle('Pressure CUDA Graph retest | median and full trial range')
    for ext in ('png', 'svg'):
        fig.savefig(root / ('wall_time.' + ext), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, mode in zip(axes, ('solver', 'pipeline')):
        rows = sorted((r for r in data if r['mode'] == mode), key=lambda r: r['grid'])
        for i, r in enumerate(rows):
            ratios = r['paired_wall_speedups']
            ax.scatter([i] * len(ratios), ratios, color='#5E35B1', alpha=.75,
                       label='Adjacent A/B process pairs' if i == 0 else None)
            ratio = r['statistics']['completed_run_wall']['speedup']
            ax.scatter(i + .13, ratio, marker='D', color='#F57C00',
                       label='Ratio of median trial times' if i == 0 else None)
            ax.annotate(f'{ratio:.2f}x', (i + .13, ratio), xytext=(5, 6), textcoords='offset points')
        ax.axhline(1, color='gray', linestyle='--')
        ax.set_xticks(range(len(rows)), [f'{r["grid"]} x {r["grid"]}' for r in rows])
        ax.set_ylabel('A wall time / B wall time (higher is faster)')
        ax.set_title(mode.capitalize())
        ax.grid(axis='y', alpha=.2)
        ax.margins(y=.25, x=.25)
    axes[0].legend(fontsize=9)
    fig.suptitle('Observed speedup | paired points show variability, not confidence intervals')
    for ext in ('png', 'svg'):
        fig.savefig(root / ('speedup.' + ext), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
