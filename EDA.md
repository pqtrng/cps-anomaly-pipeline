# EDA — HAI 21.03 (T1.5)

Facts gathered from the Bronze layer to drive the T2 Silver/Gold Pandera schema.
Regenerate with `notebooks/eda_hai.ipynb`. This file is the source of truth T2
reads.

## Dataset shape
- Train (normal only): 921,603 rows (train1-3 concatenated)
- Test (labelled): 402,005 rows (test1-5 concatenated)
- Sensor/actuator columns: 80 (P1 / P2 / P3 / P4 = 38 / 22 / 7 / 12)
- Provenance columns added at Bronze: `source_file`, `ingested_at`

## Labels
- Present in test: `attack`, `attack_P1`, `attack_P2`, `attack_P3`
- **`attack_P4` absent** — HAI 21.03 does not label process stage 4, even though
  P4 contributes 12 sensor columns. The model may learn from P4 signals, but
  per-process evaluation in T5 cannot report a P4 detection rate.

## Missing values
- **None.** 0 columns with NaN in train, 0 in test. Silver needs no imputation.

## Schema decisions for T2

### Drop in Silver (constant in BOTH train and test — 19 columns)
Zero-variance in train AND still constant in test → no information:
```
P1_PP01AD, P1_PP01AR, P1_PP01BD, P1_PP01BR, P1_PP02D, P1_PP02R, P1_STSP,
P2_ASD, P2_AutoGO, P2_MSD, P2_ManualGO, P2_RTR, P2_TripEx,
P2_VTR01, P2_VTR02, P2_VTR03, P2_VTR04, P3_LH, P3_LL
```

### KEEP despite being constant in train (3 columns — attack signal)
Constant in train (normal) but VARY in test. The variation coincides with
attacks (safety actuators changing state), so these are high-value features,
NOT noise. Dropping them would discard the strongest discrete signals:
```
P1_PCV02D   (train=12 const;  test varies)
P2_Emerg    (train=0 always;  test 0.14% =1  -> emergency asserted under attack)
P2_OnOff    (train=1 always;  test flips to 0 -> device off, abnormal)
```

### Discrete / binary actuators (small unique count) — keep, treat as discrete
- `P4_HT_PS` (2 values: 0/10), `P4_ST_PS` (2 values: 0/50)
- Plus the 3 kept above once they enter the feature set.
- Many int64 actuator columns vs float64 analog sensors — schema should type
  these distinctly, not coerce everything to float.

### Continuous sensors — require scaling before the autoencoder
Value ranges differ by orders of magnitude (e.g. `P2_CO_rpm` ~54,000 vs
`P4_ST_FD` ~0.05). Without normalisation the large-magnitude columns dominate
reconstruction loss. Standardise (fit scaler on train only) in T2/T4.

## Class balance (test) — drives T5 metric choice
| label | attack ratio | counts (0 / 1) |
|-------|--------------|----------------|
| attack | 2.23% | 393,058 / 8,947 |
| attack_P1 | 1.75% | 394,968 / 7,037 |
| attack_P2 | 0.46% | 400,141 / 1,864 |
| attack_P3 | 0.24% | 401,047 / 958 |

Heavy imbalance: predicting "normal" everywhere already scores 97.8% accuracy.
**Accuracy is meaningless here.** T5 must report per-attack precision / recall /
F1 and detection lead time — not accuracy. (Point-adjust critique also in T5.)

## Distribution shift (train normal vs test)
The extreme z_shift values (>1e6) are artefacts of train_std=0 for the three
constant-in-train columns above, not real shift. Real shifts to watch (sources
of autoencoder false alarms; account for when setting thresholds in T4):
```
P1_PCV02Z  z=1.56     P2_VYT03  z=0.98     P2_VYT02  z=0.61
P2_VT01    z=0.54     P2_SIT01/02 z=0.29   P2_VXT03  z=0.27
```

## Notes
- Bronze is raw-as-ingested (no cleaning). All decisions above apply at the
  Silver layer in T2, never retroactively to Bronze.
- Scaler is fit on train only, then applied to test — never fit on test (leakage).