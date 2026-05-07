# Method E — Vessel-Collision-Aware Aneurysm Generation

**Status:** experimental fork of methods/{C,D}.  No existing files are modified.

## Idee

Bisher ist die Aneurysma-Generierung nur **bedingt** auf das Ostium und die
umliegenden Gefäßpunkte (`vessel_pts`) — der Conditioner sieht sie zwar, aber
es gibt keine **explizite Strafe**, wenn der generierte Sac in einen
benachbarten Vessel hineinwächst.

Diese Experimentreihe fügt eine zusätzliche Loss
`VesselCollisionLoss` ein:

```
L_total = L_recon + w_collision · mean( relu(clearance − d_min(v_sac, vessel_pts))² )
```

mit `d_min` = Distanz jedes Sac-Vertex zum nächsten Punkt in `vessel_pts`,
beide Mengen im **gleichen `ghd_local` Frame** (kein Transform nötig).
Ostium-Ring-Vertices werden vom Sac-Mask ausgeschlossen, damit sie an die
Ostium-Punkte herangehen dürfen.

Variante (a) hinge-clearance ist als Startpunkt implementiert. Variante (b)
signed-penetration über die Ostium-Ebene kann später als zweiter Term mit
eigenem Gewicht ergänzt werden, ohne diesen Code zu brechen.

## Ordner

```
methods/E_collision/
├── __init__.py
├── collision_loss.py      ← VesselCollisionLoss + scheduler
├── D/train.py             ← Fork von methods/D_vq_transformer/train.py
├── C/train.py             ← Fork von methods/C_fsq_ar/train.py
├── run.sh
└── README.md
```

Outputs landen unter `checkpoints/methods/E_collision/{D,C}/`.
**Keine** Datei in `methods/A,C,D` oder `methods/_common` oder `models/` wird
verändert.

## Ausführen

```bash
# nur D
bash methods/E_collision/run.sh 1 D

# nur C
bash methods/E_collision/run.sh 1 C

# beide
bash methods/E_collision/run.sh 1 both

# Hyperparams überschreiben
W_COL=100 CLEAR=0.06 PHASE=300 RAMP=300 bash methods/E_collision/run.sh 1 D
```

Wichtige Defaults:
- `--w_collision 50.0`
- `--collision_clearance 0.04` (in `ghd_local` Einheiten; typische Sac-Radien
  liegen dort bei ~0.10–0.30, also ~10–40 % davon als Mindestabstand)
- `--collision_phase_in 200` Epochs (erst sauber rekonstruieren lernen)
- `--collision_ramp 200`

Der Trainer loggt zusätzlich:
- `col` — die unweighted Collision-Loss
- `viol_frac` — Anteil der Sac-Verts, deren `d_min < clearance`

`val_viol` sollte mit der Zeit klar unter dem Wert eines Baselines (gleicher
Seed, `--w_collision 0`) liegen.

## Methode A (PCA + FlowMatching) — geplant, noch nicht scaffolded

A trainiert nicht Reconstruction direkt, sondern ein Velocity-Field. Die
sauberste Integration wäre:

1. Pro Training-Step: Conditional Sample mit wenigen ODE-Schritten ziehen.
2. Decode (PCA-Inverse + GHD) → `pred_verts`.
3. `VesselCollisionLoss` darauf anwenden, Gradienten zurück durch Sample +
   Velocity-Net (differentiable solver) propagieren.

Das ist deutlich invasiver als D/C und betrifft `train_vessel_pca_flow_matching.py`.
Vorschlag: erst D und C laufen lassen, validieren, danach A separat angehen.

## Validierung / Visualisierung

Bestehende Tools funktionieren, sobald die Checkpoints unter dem neuen
`save_root` liegen — z. B.

```bash
python methods/visualize_val_samples.py \
  --ckpt checkpoints/methods/E_collision/D/<META>/best.pt \
  --out_dir checkpoints/methods/_viz_realcsv_E/samples
```

Für Attach + Compare: `tools/attach_aneurysm_to_healthy.py` mit dem neuen
Checkpoint als `--ckpt` (gleiches `payload`-Schema wie D-Baseline).
