import React from 'react';
import {
  BarChart3,
  ShieldCheck,
  Layers,
  Trees,
  Activity,
  Database,
  Info,
  TrendingDown,
} from 'lucide-react';

/**
 * EvaluationPanel — benchmark research figures.
 *
 * Reads /api/metrics, which serves the raw artefacts produced by the
 * evaluation scripts:
 *   stage1 = outputs/evaluation_metrics.json
 *            { iou, dice, precision, recall, accuracy, threshold, total_samples, best_train_epoch }
 *   stage2 = outputs/temporal/temporal_test_results.json
 *            { global_iou, global_dice, global_precision, global_recall, global_accuracy,
 *              tp, fp, fn, tn, ... }
 *
 * Earlier code read `val_iou` / `val_dice`, which never exist in that payload,
 * so the dashboard silently rendered hardcoded fallbacks. Both schemas are
 * accepted here, but the real keys now take priority.
 */

const RECORDED = {
  s1: { iou: 0.8386, dice: 0.9122, precision: 0.9221, recall: 0.9025, accuracy: 0.9093 },
  s2: {
    iou: 0.3828,
    dice: 0.5536,
    precision: 0.6412,
    recall: 0.4871,
    accuracy: 0.9632,
    tp: 49330,
    fp: 27602,
    fn: 51944,
    tn: 2033812,
  },
};

const S2_VAL_IOU = 0.5043;
const PCC_IOU = 0.0026;

function pick(...values) {
  const found = values.find((v) => typeof v === 'number' && Number.isFinite(v));
  return found;
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function Metric({ label, value, tone = 'cyan', suffix }) {
  return (
    <div className={`metric metric--${tone}`}>
      <span className="metric__value">
        {value}
        {suffix ? <span className="stat__unit">{suffix}</span> : null}
      </span>
      <span className="metric__label">{label}</span>
    </div>
  );
}

export default function EvaluationPanel({ analytics }) {
  const stage1 = analytics?.stage1;
  const stage2 = analytics?.stage2;

  const s1 = {
    iou: pick(stage1?.iou, stage1?.val_iou, RECORDED.s1.iou),
    dice: pick(stage1?.dice, stage1?.val_dice, RECORDED.s1.dice),
    precision: pick(stage1?.precision, stage1?.val_precision, RECORDED.s1.precision),
    recall: pick(stage1?.recall, stage1?.val_recall, RECORDED.s1.recall),
    accuracy: pick(stage1?.accuracy, stage1?.val_accuracy, RECORDED.s1.accuracy),
  };

  const s2 = {
    iou: pick(stage2?.global_iou, RECORDED.s2.iou),
    dice: pick(stage2?.global_dice, RECORDED.s2.dice),
    precision: pick(stage2?.global_precision, RECORDED.s2.precision),
    recall: pick(stage2?.global_recall, RECORDED.s2.recall),
    accuracy: pick(stage2?.global_accuracy, RECORDED.s2.accuracy),
    tp: pick(stage2?.tp, RECORDED.s2.tp),
    fp: pick(stage2?.fp, RECORDED.s2.fp),
    fn: pick(stage2?.fn, RECORDED.s2.fn),
    tn: pick(stage2?.tn, RECORDED.s2.tn),
  };

  const live = Boolean(stage1 || stage2);

  return (
    <div className="eval">
      {/* Benchmark summary strip */}
      <div className="bench-row">
        <span className="bench-row__label">
          <BarChart3 size={11} style={{ verticalAlign: '-1px', marginRight: 5 }} />
          Benchmark research — separate from live inference
        </span>
        <span className="bench-badge" title="Stage 1 U-Net validation IoU (15 images)">
          <Trees size={10} /> Forest U-Net <strong>{s1.iou.toFixed(4)}</strong>
        </span>
        <span className="bench-badge bench-badge--warn" title="Stage 2 Temporal U-Net best validation IoU">
          <Layers size={10} /> Temporal val <strong>{S2_VAL_IOU.toFixed(4)}</strong>
        </span>
        <span className="bench-badge bench-badge--warn" title="Stage 2 held-out test IoU (33 pairs)">
          <Layers size={10} /> Temporal test <strong>{s2.iou.toFixed(4)}</strong>
        </span>
      </div>

      {!live && (
        <div className="note note--info">
          <Info size={12} />
          <span>
            Live metrics endpoint unavailable — showing the recorded benchmark values from the evaluation artefacts.
          </span>
        </div>
      )}

      <div className="eval__grid">
        {/* Stage 1 */}
        <section className="eval-card">
          <div className="eval-card__head">
            <div>
              <div className="eval-card__title">Stage 1 · Forest Canopy Segmentation</div>
              <div className="eval-card__sub">
                U-Net · 7,763,041 parameters · validation split, 15 images · τ 0.5
              </div>
            </div>
            <span className="badge badge--green">
              <ShieldCheck size={10} /> Validation
            </span>
          </div>

          <div>
            <div className="slider-row" style={{ marginBottom: 4 }}>
              <span>Mean IoU</span>
              <span className="slider-value">{s1.iou.toFixed(4)}</span>
            </div>
            <div className="progress">
              <div className="progress__fill progress__fill--green" style={{ width: pct(s1.iou) }} />
            </div>
          </div>

          <div className="eval-metrics">
            <Metric label="Dice" value={s1.dice.toFixed(4)} tone="green" />
            <Metric label="Precision" value={s1.precision.toFixed(4)} tone="green" />
            <Metric label="Recall" value={s1.recall.toFixed(4)} tone="green" />
            <Metric label="Accuracy" value={s1.accuracy.toFixed(4)} tone="cyan" />
          </div>

          <div className="note">
            <Database size={11} />
            <span>
              Trained on 30 images; best checkpoint at epoch {stage1?.best_train_epoch ?? 5}. The small validation
              split means this score carries wide confidence intervals.
            </span>
          </div>
        </section>

        {/* Stage 2 */}
        <section className="eval-card">
          <div className="eval-card__head">
            <div>
              <div className="eval-card__title">Stage 2 · Temporal Change Detection</div>
              <div className="eval-card__sub">
                Temporal U-Net · 9-channel input · held-out test, 33 pairs · τ 0.5
              </div>
            </div>
            <span className="badge badge--amber">
              <Activity size={10} /> Held-out test
            </span>
          </div>

          <div>
            <div className="slider-row" style={{ marginBottom: 4 }}>
              <span>Global IoU</span>
              <span className="slider-value">{s2.iou.toFixed(4)}</span>
            </div>
            <div className="progress">
              <div className="progress__fill progress__fill--red" style={{ width: pct(s2.iou) }} />
            </div>
          </div>

          <div className="eval-metrics">
            <Metric label="Dice" value={s2.dice.toFixed(4)} tone="red" />
            <Metric label="Precision" value={s2.precision.toFixed(4)} tone="red" />
            <Metric label="Recall" value={s2.recall.toFixed(4)} tone="red" />
            <Metric label="Accuracy" value={s2.accuracy.toFixed(4)} tone="cyan" />
          </div>

          <div className="confusion">
            <div className="confusion__cell">
              <div className="confusion__value" style={{ color: 'var(--emerald-bright)' }}>
                {s2.tp.toLocaleString()}
              </div>
              <div className="confusion__label">True positive</div>
            </div>
            <div className="confusion__cell">
              <div className="confusion__value" style={{ color: 'var(--red-bright)' }}>
                {s2.fp.toLocaleString()}
              </div>
              <div className="confusion__label">False positive</div>
            </div>
            <div className="confusion__cell">
              <div className="confusion__value" style={{ color: 'var(--amber)' }}>
                {s2.fn.toLocaleString()}
              </div>
              <div className="confusion__label">False negative</div>
            </div>
            <div className="confusion__cell">
              <div className="confusion__value" style={{ color: 'var(--text-dim)' }}>
                {s2.tn.toLocaleString()}
              </div>
              <div className="confusion__label">True negative</div>
            </div>
          </div>

          <div className="note note--warn">
            <TrendingDown size={11} />
            <span>
              Recall {pct(s2.recall)}: roughly half of true forest loss is missed, and 96%
              pixel accuracy is inflated by the 4.7% positive class. Benchmark metrics do not transfer directly to
              live Sentinel-2 scenes.
            </span>
          </div>
        </section>
      </div>

      {/* Design comparison */}
      <section className="eval-card">
        <div className="eval-card__head">
          <div>
            <div className="eval-card__title">Why direct temporal modelling</div>
            <div className="eval-card__sub">
              Post-classification comparison (PCC) vs. the 9-channel Temporal U-Net
            </div>
          </div>
          <span className="badge badge--dim">Ablation</span>
        </div>

        <div className="eval-metrics" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
          <div className="metric metric--red">
            <span className="metric__value">0.0026</span>
            <span className="metric__label">PCC IoU · subtract two masks</span>
          </div>
          <div className="metric metric--green">
            <span className="metric__value">{s2.iou.toFixed(4)}</span>
            <span className="metric__label">Temporal U-Net IoU · direct</span>
          </div>
          <div className="metric metric--cyan">
            <span className="metric__value">19.94 : 1</span>
            <span className="metric__label">Train class imbalance</span>
          </div>
          <div className="metric metric--cyan">
            <span className="metric__value">4.77%</span>
            <span className="metric__label">Positive pixel ratio</span>
          </div>
        </div>

        <div className="note">
          <Info size={11} />
          <span>
            Subtracting two independently segmented masks compounds per-image error, collapsing IoU to 0.26%. Feeding
            T1, T2 and |T1 − T2| into a single network gives the first convolution an explicit change prior. The
            reported {PCC_IOU.toFixed(4)} is the measured PCC baseline, not an estimate.
          </span>
        </div>
      </section>
    </div>
  );
}
