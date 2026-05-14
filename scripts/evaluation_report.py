"""
evaluation_report.py — Full two-phase evaluation report with charts.

Phase 1 : Same signer (model creator) — 48/50 words correct (live demo)
Phase 2 : 6 unknown signers — word counts provided manually

Run:
    venv312\\Scripts\\python.exe evaluation_report.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────────
ROOT         = Path(".")
PERSONAL_DIR = ROOT / "data" / "personal_features"
FEATURES_DIR = ROOT / "data" / "extracted_features"
MODEL_PATH   = ROOT / "models" / "isl_model_solo.keras"
OUT_DIR      = ROOT / "evaluation_charts"
OUT_DIR.mkdir(exist_ok=True)

# Phase 1: same signer live demo result
PHASE1_CORRECT = 48
PHASE1_TOTAL   = 50

# Phase 2: 6 unknown signers (correct words out of 50)
PHASE2_RESULTS = {
    "Person 1": 45,
    "Person 2": 42,
    "Person 3": 46,
    "Person 4": 47,
    "Person 5": 41,
    "Person 6": 43,
}

COLORS = {
    "blue":   "#4C9BE8",
    "green":  "#2ECC71",
    "orange": "#E67E22",
    "red":    "#E74C3C",
    "purple": "#9B59B6",
    "teal":   "#1ABC9C",
    "gray":   "#95A5A6",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def multiclass_metrics(correct: int, total: int) -> dict:
    """
    Compute macro-averaged metrics for a multi-class test
    given only the number of correct predictions.
    """
    wrong     = total - correct
    accuracy  = correct / total
    # Per-class: TP=1,FP=0,FN=0 for correct; TP=0,FP=1,FN=1 for wrong
    precision = correct / total          # macro avg
    recall    = correct / total
    f1        = correct / total
    tp        = correct
    fp        = wrong
    fn        = wrong
    tn        = (total - 1) * correct + (total - 2) * wrong  # sum over all classes
    tn_avg    = tn / total
    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=round(tn_avg, 1),
                correct=correct, total=total)


# ── Load model test-set predictions (Phase 1 — held-out set) ─────────────────

def load_test_preds():
    labels       = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    label_to_idx = {name: i for i, name in enumerate(labels)}

    X_orig, y_orig = [], []
    for word_dir in sorted(PERSONAL_DIR.iterdir()):
        if not word_dir.is_dir(): continue
        word = word_dir.name
        if word not in label_to_idx: continue
        for f in sorted(word_dir.glob("seq_*.npy")):
            arr = np.load(str(f))
            if arr.shape != (30, 378): continue
            X_orig.append(arr)
            y_orig.append(label_to_idx[word])

    X_orig = np.array(X_orig, dtype=np.float32)
    y_orig = np.array(y_orig, dtype=np.int32)
    counts = np.bincount(y_orig, minlength=len(labels))
    multi_mask = np.isin(y_orig, np.where(counts > 1)[0])
    X_m, y_m = X_orig[multi_mask], y_orig[multi_mask]
    _, X_test, _, y_test = train_test_split(
        X_m, y_m, test_size=0.20, random_state=42, stratify=y_m)

    print("Loading model ...")
    model  = load_model(str(MODEL_PATH))
    probs  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    test_ids   = sorted(set(y_test.tolist()))
    test_names = [labels[i] for i in test_ids]
    return y_test, y_pred, test_ids, test_names, labels


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Confusion Matrix (held-out test set)
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrix(y_test, y_pred, test_ids, test_names):
    cm = confusion_matrix(y_test, y_pred, labels=test_ids)
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_names, yticklabels=test_names,
                linewidths=0.4, linecolor="#cccccc",
                cbar_kws={"shrink": 0.55})
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, labelpad=12)
    ax.set_title("Confusion Matrix — Phase 1 Held-Out Test Set\n(Same Signer, 200 Samples)",
                 fontsize=15, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = OUT_DIR / "chart1_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Phase 1: Per-Class Precision / Recall / F1 (bar)
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase1_per_class(y_test, y_pred, test_ids, test_names):
    report     = classification_report(y_test, y_pred, labels=test_ids,
                                       target_names=test_names, output_dict=True)
    precisions = [report[n]["precision"] for n in test_names]
    recalls    = [report[n]["recall"]    for n in test_names]
    f1s        = [report[n]["f1-score"]  for n in test_names]

    x     = np.arange(len(test_names))
    width = 0.28
    fig, ax = plt.subplots(figsize=(24, 7))
    ax.bar(x - width, precisions, width, label="Precision", color=COLORS["blue"])
    ax.bar(x,         f1s,        width, label="F1-Score",  color=COLORS["green"])
    ax.bar(x + width, recalls,    width, label="Recall",    color=COLORS["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Phase 1 — Per-Class Precision / F1-Score / Recall (Held-Out Test Set)",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    path = OUT_DIR / "chart2_phase1_per_class.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Phase 1 Summary Metrics (TP/TN/FP/FN + scores) — table style
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase1_summary():
    m = multiclass_metrics(PHASE1_CORRECT, PHASE1_TOTAL)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1 — Same Signer Live Demo Summary (48 / 50 words correct)",
                 fontsize=14, y=1.01)

    # Left: confusion breakdown bar
    ax = axes[0]
    labels_bar = ["True Positive\n(TP)", "False Positive\n(FP)",
                  "False Negative\n(FN)", "True Negative\n(TN avg/class)"]
    values = [m["tp"], m["fp"], m["fn"], m["tn"]]
    colors = [COLORS["green"], COLORS["red"], COLORS["orange"], COLORS["blue"]]
    bars = ax.bar(labels_bar, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("TP / FP / FN / TN", fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)

    # Right: metrics bar
    ax2 = axes[1]
    metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
    bar_colors    = [COLORS["teal"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
    bars2 = ax2.bar(metric_names, metric_values, color=bar_colors,
                    width=0.5, edgecolor="white")
    for bar, val in zip(bars2, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Accuracy / Precision / Recall / F1", fontsize=12)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = OUT_DIR / "chart3_phase1_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Phase 2: Per-Person Accuracy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase2_accuracy():
    names    = list(PHASE2_RESULTS.keys())
    corrects = list(PHASE2_RESULTS.values())
    accs     = [c / 50 * 100 for c in corrects]
    avg_acc  = np.mean(accs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [COLORS["blue"] if a >= avg_acc else COLORS["orange"] for a in accs]
    bars = ax.bar(names, accs, color=bar_colors, edgecolor="white", width=0.55)
    for bar, c, a in zip(bars, corrects, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{c}/50\n({a:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(avg_acc, color=COLORS["red"], linestyle="--", linewidth=1.5,
               label=f"Average: {avg_acc:.1f}%")
    ax.axhline(PHASE1_CORRECT/PHASE1_TOTAL*100, color=COLORS["green"],
               linestyle="--", linewidth=1.5,
               label=f"Phase 1 (same signer): {PHASE1_CORRECT/PHASE1_TOTAL*100:.1f}%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Phase 2 — Accuracy per Unknown Signer (out of 50 words)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = OUT_DIR / "chart4_phase2_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Phase 2: Grouped Metrics per Person
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase2_metrics():
    names   = list(PHASE2_RESULTS.keys())
    metrics = {n: multiclass_metrics(c, 50) for n, c in PHASE2_RESULTS.items()}

    accs  = [metrics[n]["accuracy"]  for n in names]
    precs = [metrics[n]["precision"] for n in names]
    recs  = [metrics[n]["recall"]    for n in names]
    f1s   = [metrics[n]["f1"]        for n in names]

    x     = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*width, accs,  width, label="Accuracy",  color=COLORS["teal"])
    ax.bar(x - 0.5*width, precs, width, label="Precision", color=COLORS["blue"])
    ax.bar(x + 0.5*width, recs,  width, label="Recall",    color=COLORS["orange"])
    ax.bar(x + 1.5*width, f1s,   width, label="F1-Score",  color=COLORS["green"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Phase 2 — Accuracy / Precision / Recall / F1 per Unknown Signer",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    for i, n in enumerate(names):
        ax.text(i, accs[i] + 0.012, f"{accs[i]:.0%}", ha="center",
                fontsize=7, color=COLORS["teal"])
    plt.tight_layout()
    path = OUT_DIR / "chart5_phase2_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Phase 2: TP / FP / FN per Person
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase2_tp_fp_fn():
    names   = list(PHASE2_RESULTS.keys())
    metrics = {n: multiclass_metrics(c, 50) for n, c in PHASE2_RESULTS.items()}
    tps = [metrics[n]["tp"] for n in names]
    fps = [metrics[n]["fp"] for n in names]
    fns = [metrics[n]["fn"] for n in names]

    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, tps, width, label="True Positive (TP)",  color=COLORS["green"])
    ax.bar(x,         fps, width, label="False Positive (FP)", color=COLORS["red"])
    ax.bar(x + width, fns, width, label="False Negative (FN)", color=COLORS["orange"])
    for i in range(len(names)):
        ax.text(i - width, tps[i] + 0.2, str(tps[i]), ha="center", fontsize=9, fontweight="bold")
        ax.text(i,         fps[i] + 0.2, str(fps[i]), ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width, fns[i] + 0.2, str(fns[i]), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Count (out of 50 words)", fontsize=11)
    ax.set_title("Phase 2 — TP / FP / FN per Unknown Signer", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = OUT_DIR / "chart6_phase2_tp_fp_fn.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Macro Average: Phase 1 vs Phase 2 comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase_comparison():
    p1 = multiclass_metrics(PHASE1_CORRECT, PHASE1_TOTAL)
    all_metrics = [multiclass_metrics(c, 50) for c in PHASE2_RESULTS.values()]
    p2_macro = {
        "accuracy":  np.mean([m["accuracy"]  for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall":    np.mean([m["recall"]    for m in all_metrics]),
        "f1":        np.mean([m["f1"]        for m in all_metrics]),
    }

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    p1_vals = [p1["accuracy"], p1["precision"], p1["recall"], p1["f1"]]
    p2_vals = [p2_macro["accuracy"], p2_macro["precision"],
               p2_macro["recall"],   p2_macro["f1"]]

    x     = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, p1_vals, width, label="Phase 1 (Same Signer)",
                color=COLORS["green"], edgecolor="white")
    b2 = ax.bar(x + width/2, p2_vals, width, label="Phase 2 (Unknown Signers — Macro Avg)",
                color=COLORS["blue"], edgecolor="white")
    for bar, val in zip(b1, p1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(b2, p2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2 — Macro Average Metrics Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    path = OUT_DIR / "chart7_phase_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Phase 2 Pie Charts (one per person + overall)
# ══════════════════════════════════════════════════════════════════════════════

def chart_phase2_pies():
    names    = list(PHASE2_RESULTS.keys())
    corrects = list(PHASE2_RESULTS.values())
    total_correct = sum(corrects)
    total_words   = 50 * len(names)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Phase 2 — Correct vs Incorrect Predictions per Signer", fontsize=15)
    axes = axes.flatten()

    for i, (name, correct) in enumerate(zip(names, corrects)):
        wrong = 50 - correct
        axes[i].pie([correct, wrong],
                    labels=[f"Correct ({correct})", f"Incorrect ({wrong})"],
                    colors=[COLORS["green"], COLORS["red"]],
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 10})
        axes[i].set_title(name, fontsize=12)

    # Overall pie in last slot
    wrong_total = total_words - total_correct
    axes[6].pie([total_correct, wrong_total],
                labels=[f"Correct ({total_correct})", f"Incorrect ({wrong_total})"],
                colors=[COLORS["teal"], COLORS["orange"]],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[6].set_title("Phase 2 — Overall\n(All 6 signers combined)", fontsize=11)
    axes[7].axis("off")

    plt.tight_layout()
    path = OUT_DIR / "chart8_phase2_pies.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Summary Table (printable image)
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_table():
    all_metrics = {n: multiclass_metrics(c, 50) for n, c in PHASE2_RESULTS.items()}
    p2_vals     = list(all_metrics.values())
    macro       = {k: np.mean([m[k] for m in p2_vals])
                   for k in ["accuracy","precision","recall","f1","tp","fp","fn"]}

    p1 = multiclass_metrics(PHASE1_CORRECT, PHASE1_TOTAL)

    rows = []
    rows.append(["Phase 1 — Same Signer",
                 f"{p1['correct']}/{p1['total']}",
                 f"{p1['tp']}", f"{p1['fp']}", f"{p1['fn']}",
                 f"{p1['accuracy']:.4f}", f"{p1['precision']:.4f}",
                 f"{p1['recall']:.4f}", f"{p1['f1']:.4f}"])

    for name, m in all_metrics.items():
        rows.append([name,
                     f"{m['correct']}/50",
                     f"{m['tp']}", f"{m['fp']}", f"{m['fn']}",
                     f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                     f"{m['recall']:.4f}", f"{m['f1']:.4f}"])

    rows.append(["Phase 2 — Macro Avg",
                 f"{sum(PHASE2_RESULTS.values())}/{50*6} total",
                 f"{macro['tp']:.1f}", f"{macro['fp']:.1f}", f"{macro['fn']:.1f}",
                 f"{macro['accuracy']:.4f}", f"{macro['precision']:.4f}",
                 f"{macro['recall']:.4f}", f"{macro['f1']:.4f}"])

    columns = ["Test Subject", "Correct/Total",
               "TP", "FP", "FN",
               "Accuracy", "Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)

    # Header style
    for j in range(len(columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Phase 1 row — green tint
    for j in range(len(columns)):
        tbl[1, j].set_facecolor("#D5F5E3")

    # Macro avg row — blue tint
    for j in range(len(columns)):
        tbl[len(rows), j].set_facecolor("#D6EAF8")
        tbl[len(rows), j].set_text_props(fontweight="bold")

    ax.set_title("ISL Recognition — Full Evaluation Summary",
                 fontsize=14, pad=20, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "chart9_summary_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_text_report():
    p1 = multiclass_metrics(PHASE1_CORRECT, PHASE1_TOTAL)
    all_metrics = {n: multiclass_metrics(c, 50) for n, c in PHASE2_RESULTS.items()}
    p2_vals = list(all_metrics.values())
    macro = {k: np.mean([m[k] for m in p2_vals])
             for k in ["accuracy","precision","recall","f1","tp","fp","fn"]}

    lines = []
    lines.append("ISL Recognition System — Full Evaluation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("PHASE 1 — SAME SIGNER (Live Demo)")
    lines.append("-" * 40)
    lines.append(f"  Words correct   : {p1['correct']} / {p1['total']}")
    lines.append(f"  True Positive   : {p1['tp']}")
    lines.append(f"  False Positive  : {p1['fp']}")
    lines.append(f"  False Negative  : {p1['fn']}")
    lines.append(f"  True Negative   : {p1['tn']} (macro avg per class)")
    lines.append(f"  Accuracy        : {p1['accuracy']:.4f}  ({p1['accuracy']:.2%})")
    lines.append(f"  Precision       : {p1['precision']:.4f}")
    lines.append(f"  Recall          : {p1['recall']:.4f}")
    lines.append(f"  F1-Score        : {p1['f1']:.4f}")
    lines.append("")
    lines.append("PHASE 2 — UNKNOWN SIGNERS")
    lines.append("-" * 40)
    lines.append(f"  {'Subject':<20} {'Correct':>8} {'TP':>4} {'FP':>4} {'FN':>4} "
                 f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    lines.append("  " + "-" * 72)
    for name, m in all_metrics.items():
        lines.append(f"  {name:<20} {str(m['correct'])+'/50':>8} {m['tp']:>4} "
                     f"{m['fp']:>4} {m['fn']:>4} "
                     f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                     f"{m['recall']:>8.4f} {m['f1']:>8.4f}")
    lines.append("  " + "-" * 72)
    lines.append(f"  {'Macro Average':<20} "
                 f"{sum(PHASE2_RESULTS.values())}/{50*len(PHASE2_RESULTS):>4}  "
                 f"{macro['tp']:>4.1f} {macro['fp']:>4.1f} {macro['fn']:>4.1f} "
                 f"{macro['accuracy']:>8.4f} {macro['precision']:>8.4f} "
                 f"{macro['recall']:>8.4f} {macro['f1']:>8.4f}")
    lines.append("")
    lines.append("CHARTS GENERATED")
    lines.append("-" * 40)
    for i, desc in enumerate([
        "Confusion Matrix (Phase 1 held-out test set)",
        "Per-Class Precision / Recall / F1 (Phase 1)",
        "Phase 1 Summary — TP/FP/FN + Scores",
        "Phase 2 Accuracy per Unknown Signer",
        "Phase 2 Grouped Metrics per Person",
        "Phase 2 TP / FP / FN per Person",
        "Phase 1 vs Phase 2 Comparison",
        "Phase 2 Pie Charts (per person + overall)",
        "Full Summary Table",
    ], 1):
        lines.append(f"  chart{i}_*.png — {desc}")

    text = "\n".join(lines)
    path = OUT_DIR / "evaluation_report.txt"
    path.write_text(text)
    print(f"  Saved {path.name}")
    print()
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading test predictions ...")
    y_test, y_pred, test_ids, test_names, labels = load_test_preds()

    print("\nGenerating charts ...")
    chart_confusion_matrix(y_test, y_pred, test_ids, test_names)
    chart_phase1_per_class(y_test, y_pred, test_ids, test_names)
    chart_phase1_summary()
    chart_phase2_accuracy()
    chart_phase2_metrics()
    chart_phase2_tp_fp_fn()
    chart_phase_comparison()
    chart_phase2_pies()
    chart_summary_table()
    save_text_report()

    print(f"\nAll outputs saved to: {OUT_DIR}/")
