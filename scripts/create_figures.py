"""Generate three publication-quality figures for the README."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Consistent style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
})

# Color palette
C_V2 = '#5B8DBE'       # steel blue
C_V3 = '#E07B54'       # warm orange
C_CRF = '#5B8DBE'      # steel blue
C_BERT = '#E07B54'     # warm orange
C_REMAIN = '#4A7C59'   # forest green
C_REMOVED = '#D4A574'  # tan/sand

OUTDIR = 'results/figures'

# ─────────────────────────────────────────────
# Figure 1: Ablation V2 vs V3
# ─────────────────────────────────────────────
def fig1():
    variants = ['Main', 'A\n(No Noise)', 'B\n(No Oral)', 'C\n(No Partial)']
    v2_clean = [0.739, 0.754, 0.679, 0.496]
    v3_clean = [0.793, 0.899, 0.755, 0.438]
    v2_noisy = [0.625, 0.476, 0.250, 0.067]
    v3_noisy = [0.540, 0.540, 0.479, 0.095]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(variants))
    w = 0.32

    for ax, v2, v3, title in [
        (ax1, v2_clean, v3_clean, 'Clean Strict F1 (n=41)'),
        (ax2, v2_noisy, v3_noisy, 'Noisy Strict F1 (V2: n=13, V3: n=74)')
    ]:
        bars_v2 = ax.bar(x - w/2, v2, w, color=C_V2, label='V2', edgecolor='white', linewidth=0.8)
        bars_v3 = ax.bar(x + w/2, v3, w, color=C_V3, label='V3', edgecolor='white', linewidth=0.8)
        for bar in bars_v2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
        for bar in bars_v3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Strict F1')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Annotation on clean panel — Variant A V3 bar
    ax1.annotate(
        'No noise outperforms Main\n— reverses V2 finding',
        xy=(1 + w/2, 0.899), xytext=(2.2, 0.96),
        fontsize=10, fontweight='bold', color='#C0392B',
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2),
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', edgecolor='#C0392B', alpha=0.9)
    )

    fig.suptitle('Ablation Study: V2 vs V3 Comparison', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/ablation_v2_vs_v3.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved ablation_v2_vs_v3.png')


# ─────────────────────────────────────────────
# Figure 2: Error Reduction Waterfall
# ─────────────────────────────────────────────
def fig2():
    steps = [
        'BERT-Small\nstandalone',
        '+ Rules routing\n(CONTRACT_ID)',
        '+ Oral-format\nregex',
        '+ Min span\nfilter',
        '+ Self-correction\nfilter',
        'Final\nensemble',
    ]
    remaining = [33, 20, 16, 13, 12, 12]
    removed   = [ 0, 13,  4,  3,  1,  0]
    labels    = ['', 'CONTRACT_ID → rules', 'oral CONTRACT_ID\npatterns',
                 'single-char\nfalse alarms', "'david'\nfalse alarm", '']

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(steps))[::-1]  # top to bottom
    bar_h = 0.55

    for i in range(len(steps)):
        # Remaining errors bar
        color = C_REMAIN if i < len(steps) - 1 else '#2E6B4A'
        ax.barh(y[i], remaining[i], height=bar_h, color=color, edgecolor='white', linewidth=0.8)
        # Remaining count label
        ax.text(remaining[i] / 2, y[i], str(remaining[i]),
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        if removed[i] > 0:
            # Removed segment (lighter, stacked after remaining)
            ax.barh(y[i], removed[i], left=remaining[i], height=bar_h,
                    color=C_REMOVED, edgecolor='white', linewidth=0.8, alpha=0.85)
            # Removed count inside segment
            ax.text(remaining[i] + removed[i] / 2, y[i],
                    f'−{removed[i]}', ha='center', va='center', fontsize=12, fontweight='bold', color='#8B4513')
            # Label to the right
            ax.text(remaining[i] + removed[i] + 0.5, y[i], labels[i],
                    ha='left', va='center', fontsize=10, color='#666666')

    ax.set_yticks(y)
    ax.set_yticklabels(steps)
    ax.set_xlabel('Number of Errors', fontsize=14)
    ax.set_xlim(0, 40)
    ax.set_title('Ensemble Error Reduction Waterfall', fontsize=18, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # "64% error reduction" annotation
    ax.text(20, y[-1] - 0.8, '64% error reduction (33 → 12)',
            fontsize=14, fontweight='bold', color='#2E6B4A',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E6B4A'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_REMAIN, label='Remaining errors'),
        mpatches.Patch(facecolor=C_REMOVED, alpha=0.85, label='Errors removed by component'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/ensemble_error_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved ensemble_error_waterfall.png')


# ─────────────────────────────────────────────
# Figure 3: CRF vs BERT Per-Entity
# ─────────────────────────────────────────────
def fig3():
    entities = ['CONTRACT_ID', 'EMAIL', 'ISSUE_DATE', 'NAME', 'PRODUCT']
    crf_clean  = [0.933, 0.818, 0.897, 0.583, 0.857]
    bert_clean = [0.581, 0.818, 0.824, 0.889, 0.833]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [3, 1.2]})

    # Left panel: per-entity clean
    x = np.arange(len(entities))
    w = 0.32
    ax1.bar(x - w/2, crf_clean,  w, color=C_CRF,  label='CRF', edgecolor='white', linewidth=0.8)
    ax1.bar(x + w/2, bert_clean, w, color=C_BERT, label='BERT-Small', edgecolor='white', linewidth=0.8)

    # Value labels and winner stars
    for i in range(len(entities)):
        crf_val, bert_val = crf_clean[i], bert_clean[i]
        # CRF bar label
        ax1.text(x[i] - w/2, crf_val + 0.015, f'{crf_val:.3f}',
                ha='center', va='bottom', fontsize=9.5)
        # BERT bar label
        ax1.text(x[i] + w/2, bert_val + 0.015, f'{bert_val:.3f}',
                ha='center', va='bottom', fontsize=9.5)
        # Star on winner
        if crf_val > bert_val:
            ax1.plot(x[i] - w/2, crf_val + 0.06, '*', markersize=14, color='gold',
                    markeredgecolor='#B8860B', markeredgewidth=1)
        elif bert_val > crf_val:
            ax1.plot(x[i] + w/2, bert_val + 0.06, '*', markersize=14, color='gold',
                    markeredgecolor='#B8860B', markeredgewidth=1)

    # Annotations
    ax1.annotate("BERT's only\ndecisive win", xy=(3 + w/2, 0.889), xytext=(3.7, 0.65),
                fontsize=10, fontweight='bold', color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2),
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', edgecolor='#C0392B', alpha=0.9))

    ax1.annotate("CRF: P=1.000\nR=0.812", xy=(2 - w/2, 0.897), xytext=(0.8, 0.65),
                fontsize=10, fontweight='bold', color='#2471A3',
                arrowprops=dict(arrowstyle='->', color='#2471A3', lw=2),
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8', edgecolor='#2471A3', alpha=0.9))

    ax1.set_xticks(x)
    ax1.set_xticklabels(entities, fontsize=11)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel('Strict F1')
    ax1.set_title('Clean Strict F1 by Entity Type', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: overall noisy
    noisy_labels = ['CRF', 'BERT-Small']
    noisy_vals = [0.354, 0.540]
    colors = [C_CRF, C_BERT]
    bars = ax2.bar(noisy_labels, noisy_vals, width=0.5, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, noisy_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel('Strict F1')
    ax2.set_title('Overall Noisy\nStrict F1', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('CRF vs BERT-Small: Complementary Strengths', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/crf_vs_bert_entity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved crf_vs_bert_entity.png')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print('All figures generated.')
