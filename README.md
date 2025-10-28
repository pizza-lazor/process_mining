# Process Mining Explorer

Interactive Celonis-style process mining desktop application powered by [pm4py](https://pm4py.fit.fraunhofer.de/) and [PyQt](https://riverbankcomputing.com/software/pyqt/).

This project lets you load event logs (CSV or XES), apply flexible filters, discover process models, inspect variants, analyse time-series performance metrics, and run conformance checking — all from a local PyQt GUI.

## Features
- Explore the bundled order-to-cash sample log or load your own CSV/XES event logs.
- Auto-detect or manually map case/activity/timestamp columns for CSV uploads.
- Filter by cases, activities, time windows, or custom attributes before running analyses.
- Discover Petri nets with the Inductive Miner, toggle frequency/performance overlays, explore an interactive graph, and export PNML/XES snapshots.
- Visualise frequency and performance Directly-Follows Graphs.
- Enjoy interactive pyqtgraph dashboards for throughput histograms, time-series trends, variant coverage, top resources, and activity rework.
- Run Petri-net alignments to monitor conformance and fitness scores.

## Environment Setup

### Cross-platform bootstrap
```bash
python scripts/bootstrap_env.py
```
Use `--force` to recreate the virtual environment, and `--python` to target a specific interpreter. The script works on Linux, macOS, and Windows.

### Quick start (venv)
```bash
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Launch the PyQt app:
```bash
python pyqt_app.py
```

### Conda environment
An `environment.yml` is included for Conda users:
```bash
conda env create -f environment.yml
conda activate pm4py-gui
python pyqt_app.py
```

### PowerShell helper
On Windows PowerShell, run:
```powershell
.\scripts\setup-env.ps1
```
Add `-Force` to recreate the virtual environment. The script delegates to the cross-platform bootstrap helper so behaviour matches the command above.

## Usage Tips
- Use the **Load Event Log** panel to choose the sample log or open your own CSV/XES files.
- For CSV uploads, map columns to the canonical `case_id`, `activity`, and `timestamp` fields using the mapping dialog.
- Adjust filters (cases, activities, time range, attributes) to focus on relevant slices before analysing.
- Export the currently filtered log as XES or the discovered Petri net as PNML for downstream tooling.

## Repository Layout
```
.
├── app/
│   ├── __init__.py                # Package init
│   ├── log_filters.py             # Helper functions to slice logs
│   ├── log_loader.py              # CSV/XES ingestion and normalisation utilities
│   └── process_analysis.py        # Process discovery, analytics, and conformance helpers
├── data/
│   └── sample_order_to_cash.csv   # Order-to-cash demo dataset
├── pyqt_app.py                    # PyQt entry point
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment definition
├── scripts/
│   └── setup-env.ps1              # PowerShell helper for venv creation
└── README.md
```

## Next Ideas
1. Add KPI customisation (e.g., lead time per variant or activity SLA breaches).
2. Persist user-uploaded logs via a lightweight backend or cloud storage.
3. Extend filters (resource hand-offs, case attributes) and support drill-down dashboards.
# process_mining
