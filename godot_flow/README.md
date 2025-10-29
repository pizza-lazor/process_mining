# Godot Flow Explorer Prototype

This folder contains a starter Godot 4 project that visualises the JSON payload exported from the PyQt application. The goal is to experiment with a game-engine driven experience (particle effects, animated edges, free-camera navigation) without disturbing the existing UI.

## Requirements

- [Godot Engine 4.2+](https://godotengine.org/download)
- Python environment for the exporter script (`scripts/export_flow_for_godot.py`)

## Workflow

1. **Export flow data**

   - From the PyQt app: open a log, then click `Export Flow Payload (JSON)` in the Process Model controls. Save the file to `godot_flow/data/flow_payload.json` (or any location of your choosing).
   - CLI alternative:

     ```bash
     source .venv/bin/activate
     python scripts/export_flow_for_godot.py \
       --input data/sample_order_to_cash.csv \
       --output godot_flow/data/flow_payload.json
     ```

   Both options auto-detect CSV columns (or consume XES files) and emit the JSON payload consumed by the Godot scene.

2. **Open the Godot project**

   - Launch Godot 4 and open the project at `godot_flow/project.godot`.
   - Ensure the scene `scenes/Main.tscn` is the main scene (already configured in `project.godot`).
   - Press ▶️ Run. The scene reads `res://data/flow_payload.json` (configure via the `payload_path` export field in the inspector) and renders a radial layout of the nodes and edges.

3. **Iterate with richer visuals**

   The provided `FlowExplorer.gd` script is intentionally simple: it positions nodes on a ring, draws neon edges, and labels activities. Extend it with:

   - Particle trails or shaders for animated hand-offs.
   - Case playback (animate a marker through the sequence).
   - Input gestures (panning/zooming, selection highlights).
   - Shader Graph/VisualShader for glow or bloom effects.

## File Overview

- `project.godot` – minimal engine configuration referencing the main scene.
- `default_env.tres` – sets a dark ambient environment.
- `scenes/Main.tscn` – root scene hosting the `FlowExplorer` Node2D and a `Camera2D`.
- `scripts/FlowExplorer.gd` – loads the JSON payload, builds a lightweight visualisation, and sketches the rendering pipeline we can enhance.
- `data/sample_flow_payload.json` – canned example so the scene runs out-of-the-box.

## Next Steps

- Replace the radial layout with a force-directed or layered algorithm (e.g., port the Python layout to Godot).
- Mirror interactions back to the PyQt app (WebSocket or gRPC bridge so filters stay in sync).
- Experiment with Godot’s `GraphNode`/`GraphEdit` or custom shaders for a richer HUD.
- Evaluate performance against larger logs; migrate to threads/GDScript coroutines for streaming updates.
