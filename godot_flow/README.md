# Godot Flow Explorer Prototype

This directory holds the experimental Godot 4 “immersive” renderer that consumes data exported from the PyQt application. The aim is to explore richer, game-inspired visuals (animated flows, bloom, camera motion) while keeping the existing desktop UX intact.

---

## 1. Architecture Overview

### Data pipeline
1. **Source log** – User loads/filters a log inside the PyQt app.
2. **Payload builder** – `app/process_analysis.build_flow_payload` converts the filtered log into a compact JSON structure (nodes, edges, metrics).
3. **Exporter** – Either:
   - `ProcessMiningApp.export_flow_payload()` (GUI button) writes the JSON, or
   - `scripts/export_flow_for_godot.py` (CLI) creates the same payload.
4. **Godot runtime** – `godot_flow/scripts/FlowExplorer.gd` reads the JSON, builds an in-memory graph, and renders it with GDScript.

```
PyQt filters --> build_flow_payload --> flow_payload.json --> Godot FlowExplorer
```

### Payload structure (excerpt)
```json
{
  "metadata": {
    "total_cases": 120,
    "kept_activities": 16,
    "max_edge_weight": 540
  },
  "nodes": [
    { "id": "__start__", "label": "Start", "frequency": 120, "cases": 120 },
    { "id": "Check Inventory", "label": "Check Inventory", "frequency": 108, "cases": 102 }
  ],
  "edges": [
    {
      "source": "Check Inventory",
      "target": "Pack Goods",
      "frequency": 104,
      "cases": 88,
      "duration_seconds": 3600.0,
      "is_rework": false
    }
  ]
}
```

---

## 2. Exporting Flow Data

### From the PyQt app
1. Load/filter a log as usual.
2. On the **Process Model** tab, click `Export Flow Payload (JSON)`.
3. Choose a location (e.g., `godot_flow/data/flow_payload.json`).

### From the command line
```bash
source .venv/bin/activate
python scripts/export_flow_for_godot.py \
  --input data/sample_order_to_cash.csv \
  --output godot_flow/data/flow_payload.json
```
- CSV columns are auto-detected (or use XES files directly).
- `--input` may be any log the Python stack can load.

---

## 3. Running the Godot Prototype

1. Install [Godot 4.x](https://godotengine.org/download) if you haven’t already.
2. Launch Godot and **Import** `godot_flow/project.godot` (or run `godot4 --editor godot_flow/project.godot`).
3. Ensure `scenes/Main.tscn` is set as the main scene (already configured).
4. Press ▶️ (F5) to run. The scene loads `res://data/flow_payload.json` by default; adjust the `payload_path` export variable in the inspector to target another file.

The current implementation:
- Positions nodes on a circular layout.
- Draws coloured, weighted edges with basic duration/rework tinting.
- Provides a `Camera2D` with mouse pan/zoom (via default Godot settings).

---

## 4. Development Notes

### Key files
| Path | Purpose |
|------|---------|
| `project.godot` | Engine config; sets main scene, window size |
| `default_env.tres` | Scene environment (dark background) |
| `scenes/Main.tscn` | Root scene: `FlowExplorerRoot` + `Camera2D` |
| `scripts/FlowExplorer.gd` | Loads JSON, lays out nodes, draws edges |
| `data/sample_flow_payload.json` | Example payload for quick smoke tests |
| `scripts/export_flow_for_godot.py` | CLI exporter (mirrors PyQt button) |

### `FlowExplorer.gd` responsibilities
- `_load_payload()` – reads JSON, normalises types, stores metadata.
- `_layout_nodes()` – currently radial placement, ready for force-directed replacement.
- `_draw_edges()` / `_draw_nodes()` – immediate-mode drawing of the graph each frame.
- `@export` properties – let designers switch payload paths or base radius from the editor.

### Style & typing
- Script is written for GDScript 2.0 (Godot 4) with explicit typing to suppress warnings.
- Any structural change to the payload should be mirrored in `build_flow_payload`.

---

## 5. Roadmap for Enhancement

| Area | Ideas |
|------|-------|
| Visual polish | Bloom, glow outlines, motion trails, ambient animation |
| Layout | Port force-directed / layered layout from Python, integrate case playback paths |
| Interaction | Hover popovers, selection lock, on-canvas filters, timeline scrubbing |
| Sync | Bi-directional channel (WebSocket/gRPC) so PyQt filters update Godot in real time |
| Deployment | Package Godot export alongside PyQt build, launch via CLI (`--payload` argument) |

See `roadmap.md` (Stage 6) for tracked milestones and future tasks.

---

## 6. Deployment Considerations

- **Desktop bundle** – export Godot project for each target OS and ship the executable + `.pck` with the PyQt installer. Launch it via `subprocess` when the user requests “Immersive Flow”.
- **Web** (optional) – export to WebAssembly, host it, and embed via QtWebEngine for an in-app experience without distributing a second executable.
- **Live updates** – use temporary files for one-shot launches or implement a persistent IPC layer for streaming updates.
- **Licensing** – Godot is MIT; no obligations beyond attribution.

---

## 7. Troubleshooting

- **“No payload path provided”** – set `payload_path` in the inspector or export a JSON file first.
- **Type warnings** – ensure Godot 4.5+ (the script relies on typed arrays/dictionaries).
- **Empty graph** – verify your JSON includes `nodes`/`edges` and matches the schema above.

---

With this workflow, design iterations can happen in Godot while analytics remain in Python. Once the visuals reach parity with expectations, we can wire a launch hook in PyQt and ship both experiences together.
