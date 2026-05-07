# Ostium GUI und Vessel-Cut im Repo

## Kurzfassung

Es gibt jetzt eine lokale Web-GUI:

- `tools/ostium_cut_gui.py`

Die GUI kann ein Healthy-Vessel-OBJ laden, per Klick einen Ostium-Mittelpunkt
setzen, die lokale Normalenrichtung uebernehmen, Radius/Elliptizitaet
anpassen, lokale Vessel-Punkte sampeln und daraus ein CVAE-kompatibles
Condition-Paket exportieren. Optional kann sie weiter einen Cut previewen und
ein geschnittenes Vessel als OBJ exportieren.

Zusaetzlich gibt es weiterhin:

1. Den Open3D-Point-Picker fuer manuelle Opening-/Ostium-Registrierung.
2. Den CLI-Prototyp `tools/attach_aneurysm_to_healthy.py`.

## 0. Moderne Ostium-Cut-GUI

Starten:

```bash
cd /path/to/SynVA-A1

conda run --no-capture-output -n unified_env python tools/ostium_cut_gui.py \
  --host 127.0.0.1 \
  --port 8050
```

Danach im Browser oeffnen:

```text
http://127.0.0.1:8050
```

### Auf einem Server ohne GUI

Die Web-GUI braucht kein Open3D-Fenster, kein `DISPLAY` und keine Desktop-
Session auf dem Server. Dash/Plotly rendert im Browser auf deinem lokalen
Rechner. Auf dem Server muss nur der Python-Webserver laufen.

Empfohlene Variante mit SSH-Port-Forwarding:

Auf deinem lokalen Rechner:

```bash
ssh -L 8050:127.0.0.1:8050 <user>@<server>
```

In der SSH-Session auf dem Server:

```bash
cd /path/to/SynVA-A1

conda run --no-capture-output -n unified_env python tools/ostium_cut_gui.py \
  --host 127.0.0.1 \
  --port 8050
```

Dann lokal im Browser oeffnen:

```text
http://127.0.0.1:8050
```

Wenn Port `8050` schon belegt ist, nimm z. B. `8051` in beiden Befehlen.

Direkt lokal auf deinem Rechner starten geht auch, aber dann muessen Repo,
Python-Umgebung und die Mesh-/NPY-Dateien lokal verfuegbar sein.

Mit einem konkreten Case vorbefuellen:

```bash
conda run --no-capture-output -n unified_env python tools/ostium_cut_gui.py \
  --host 127.0.0.1 \
  --port 8050 \
  --case aneux_C0075
```

Oder direkt mit Mesh-/Ostium-Dateien:

```bash
conda run --no-capture-output -n unified_env python tools/ostium_cut_gui.py \
  --healthy_mesh /path/to/vessel.obj \
  --ostium_centroid /path/to/centroid_ostium.npy \
  --ostium_normal /path/to/normal_vector.npy
```

In der GUI:

- `Load` laedt das Vessel.
- Klick auf einen angezeigten Vertex setzt Ostium-Zentrum und lokale Normalen.
- `Load ostium` laedt `centroid_ostium.npy` und `normal_vector.npy`.
- `Flip` dreht die Normale um.
- `Preview condition` zeigt die lokale Vessel-Punktwolke und die Ostium-Kontur.
- `Export condition` schreibt die CVAE-Eingaben:
  - `cvae_condition.npz`
  - `ostium_params.npy`
  - `vessel_pts.npy`
  - `condition_metadata.json`
- `Preview` berechnet den Cut und zeigt Boundary/entfernte Faces.
- `Export` schreibt das geschnittene Vessel und optional den JSON-Report.

Default-Outputs landen unter:

```text
/path/to/SynVA-A1/checkpoints/ostium_cut_gui/<case>/
```

### CVAE-Condition-Format

`Export condition` erzeugt:

```text
ostium_params: [8]    center_xyz + normal_xyz + radius + eccentricity
vessel_pts:    [N,3]  lokale Vessel-Oberflaechenpunkte um das Ostium
```

Das entspricht der Schnittstelle aus `models/vae_datasets_vessel.py` und
`models/vessel_conditioner.py`.

Zum Laden in Python:

```python
import numpy as np
import torch
from models.vessel_aware_cvae_ensemble import VesselAwareCVAEEnsemble

cond = np.load("/path/to/SynVA-A1/checkpoints/ostium_cut_gui/aneux_C0075/cvae_condition.npz")
ostium_params = torch.from_numpy(cond["ostium_params"]).float().unsqueeze(0)
vessel_pts = torch.from_numpy(cond["vessel_pts"]).float().unsqueeze(0)

ens = VesselAwareCVAEEnsemble(
    ["/path/to/models_best_val.pth"],
    device="cuda",
)
samples_ghd = ens.sample_ghd(vessel_pts, ostium_params, k_per_member=8)
```

Wichtig: `VesselAwareCVAEEnsemble.sample(...)` normalisiert die Inputs selbst,
solange `normalize_inputs=True` bleibt.

### Raw vs. GHD-local

Viele neue Trainingslaeufe nutzen `condition_space="ghd_local"`. Dann muessen
`ostium_params` und `vessel_pts` in genau diesem Koordinatenraum liegen. Die GUI
kann das beim Export erledigen, wenn im Bereich `GHD-local transform` diese
Pfade gesetzt sind:

- `prealign_transform.npy`
- GHD-Fitting-Checkpoint, z. B. `ghb_fitting_checkpoint.pkl`
- Canonical mesh, z. B. `/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj`

Wenn du nur den geometrischen Zustand im Prepared/Raw-Vessel-Raum exportieren
willst, lasse `Space = raw`.

## 1. Manuelle grafische Oberflaeche

Die GUI steckt hier:

- `utils/utils_registration.py`
  - `pick_points(...)`
  - nutzt `open3d.visualization.VisualizerWithEditing`
- `ghd/fitting/registration.py`
  - `RegistrationwOpeningAlignment.register_openings()`
  - ruft `u_register.pick_points(pcd)` auf
  - baut danach mit `create_opening_meshes()` ein Opening-/Ostium-Patch
  - speichert das Ergebnis mit `save_checkpoint_opa(...)` als
    `opa_checkpoint*.pkl`

Diese Oberflaeche dient zum Klicken von Punkten auf einem vorhandenen
Opening/Ostium. Sie schneidet nicht automatisch ein neues Loch aus einem
geschlossenen Vessel.

### Starten

Beispiel fuer einen einzelnen Case mit einem Ostium:

```bash
cd /path/to/SynVA-A1

conda run --no-capture-output -n unified_env python - <<'PY'
from types import SimpleNamespace
from ghd.fitting.registration import RegistrationwOpeningAlignment

args = SimpleNamespace(device="cpu")

root = "/path/to/ghd_prepared_meshes_3_aneurysm_1op_new"
case = "aneux_C0075"

reg = RegistrationwOpeningAlignment(
    args=args,
    root=root,
    target=case,
    num_op=1,
    suffix=".obj",
)

reg.load_checkpoint_opa(
    f"{root}/{case}/opa_checkpoint_manual.pkl",
    redo=True,
    auto=False,
)
PY
```

Was dann passiert:

- Es oeffnet sich ein Open3D-Fenster.
- Mit `Shift + left click` Punkte auf dem Ostium-/Opening-Rand setzen.
- Mit `Shift + right click` den letzten Punkt rueckgaengig machen.
- Mit `Q` das Fenster schliessen.
- Danach wird `opa_checkpoint_manual.pkl` im Case-Ordner geschrieben.

Praktisch reichen nicht nur drei Punkte: fuer ein brauchbares Ostium lieber
mehrere Punkte einmal um den Rand herum klicken. Die Reihenfolge ist nicht
kritisch, `register_openings()` sortiert die Punkte danach.

### Wichtig bei Remote/Servern

Open3D braucht ein echtes Display. Wenn du per SSH oder in einem Container
arbeitest, muss X11/GUI-Forwarding funktionieren, sonst erscheint kein Fenster.
Typische Symptome sind Fehler um `DISPLAY`, GLFW oder OpenGL.

## 2. Vessel schneiden: CLI-Prototyp

Das Werkzeug, das wirklich ein Ostium-Loch in ein Healthy-Vessel schneidet,
liegt hier:

- `tools/attach_aneurysm_to_healthy.py`
  - `_cut_healthy(...)` entfernt Faces im Ostium-Bereich
  - `--out_cut_mesh` schreibt das geschnittene Healthy-Vessel
  - `--out_mesh` schreibt zusaetzlich das Vessel mit angenaehtem Aneurysma

Die Pipeline ist auch in `docs/aneurysm_attach_pipeline.md` beschrieben.
Die neue Web-GUI oben ist die interaktive Variante fuer den reinen Vessel-Cut;
dieses CLI-Tool bleibt praktisch, wenn zusaetzlich ein Aneurysma angenaeht
werden soll.

### Starten

Beispiel mit Default-Pfaden fuer `aneux_C0075`:

```bash
cd /path/to/SynVA-A1

conda run --no-capture-output -n unified_env python tools/attach_aneurysm_to_healthy.py \
  --case aneux_C0075 \
  --jagged_amp 0.16 \
  --radius_scale 1.10 \
  --cut_slab 0.06 \
  --out_cut_mesh /path/to/SynVA-A1/checkpoints/attach_aneurysm/aneux_C0075/aneux_C0075_cut_vessel.obj \
  --out_mesh /path/to/SynVA-A1/checkpoints/attach_aneurysm/aneux_C0075/aneux_C0075_attached.obj \
  --out_report /path/to/SynVA-A1/checkpoints/attach_aneurysm/aneux_C0075/report.json
```

Mit `--case` erwartet das Tool standardmaessig:

- Healthy vessel:
  `/path/to/healthy_vessel/<case>_vessel_submesh_closed/<case>_vessel_submesh_closed.obj`
- Aneurysma-Mesh:
  `/path/to/prepared_meshes_3/<case>/05_submeshes/aneurysm_submesh.obj`
- Ostium-Zentrum:
  `/path/to/prepared_meshes_3/<case>/07_other/centroid_ostium.npy`
- Ostium-Normal:
  `/path/to/prepared_meshes_3/<case>/07_other/normal_vector.npy`
- Aneurysma-Labels:
  `/path/to/prepared_meshes_3/<case>/06_submesh_labels/labels_aneurysm.npy`

Wenn die Daten woanders liegen, die Pfade explizit setzen:

```bash
conda run --no-capture-output -n unified_env python tools/attach_aneurysm_to_healthy.py \
  --healthy_mesh /path/to/vessel.obj \
  --aneurysm_mesh /path/to/aneurysm.obj \
  --ostium_centroid /path/to/centroid_ostium.npy \
  --ostium_normal /path/to/normal_vector.npy \
  --aneurysm_labels /path/to/labels_aneurysm.npy \
  --cut_radius 1.5 \
  --out_cut_mesh /path/to/cut_vessel.obj \
  --out_mesh /path/to/attached.obj
```

## Welche Variante ist die richtige?

- Wenn du Punkte auf einem vorhandenen Ostium/Opening manuell markieren und
  daraus ein `opa_checkpoint*.pkl` erzeugen willst: Open3D-GUI ueber
  `RegistrationwOpeningAlignment.register_openings()`.
- Wenn du in ein Healthy-Vessel interaktiv ein Ostium-Loch schneiden willst:
  Web-GUI `tools/ostium_cut_gui.py`.
- Wenn du Cut plus Aneurysma-Stitching per Batch/CLI willst:
  `tools/attach_aneurysm_to_healthy.py` mit `--out_cut_mesh` und `--out_mesh`.
