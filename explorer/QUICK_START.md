# Capsule Explorer - Quick Start Guide

## What Changed

The Flask explorer can now:
1. Download capsule data from HuggingFace
2. Extract and preserve Jupyter notebooks
3. Display notebooks inline with rich rendering
4. Show data file listings

## How to Use

### Start the Explorer
```bash
cd explorer
./run.sh
```
Visit: http://localhost:5001

### Load a Capsule
1. Go to "Capsules" in the navbar
2. Click on any capsule (e.g., bix-1)
3. Click "Load Capsule Data" button
4. Wait for download (page auto-reloads)

### View the Notebook
1. After data loads, click "View Notebook"
2. Scroll through the rendered notebook
3. Click "Hide Notebook" to see file list

## Technical Details

### New Dependencies
- `nbformat`: Parse Jupyter notebooks
- `nbconvert`: Convert notebooks to HTML

Install via:
```bash
source .venv/bin/activate
uv pip install nbformat nbconvert
```

### File Structure After Extraction
```
explorer/cache/bix-1/
├── file1.csv           # Data files (flattened to root)
├── file2.txt
├── file3.xlsx
└── CapsuleNotebook-<uuid>/
    └── notebook_executed.ipynb  # Preserved!
```

### API Endpoints

**Load Capsule**
```
GET /api/capsule/<capsule_id>/load
Response: {success: true, files: [...], has_notebook: true}
```

**Get Notebook HTML**
```
GET /api/capsule/<capsule_id>/notebook
Response: {success: true, html: "...", notebook_name: "..."}
```

## Example Capsule

Test with: bix-1 (or any capsule ID from the list)
- Has ~4 data files
- Includes executed Jupyter notebook
- Notebook shows R-based RNA-seq analysis with DESeq2

## Troubleshooting

**"Capsule not found" error**
- Capsule may not have data files on HuggingFace
- Check that benchmark dataset loaded (startup logs)

**Notebook doesn't render**
- Check that nbconvert is installed: `pip list | grep nbconvert`
- View browser console for errors
- Check server logs for notebook path

**Dependencies missing**
```bash
cd explorer
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Implementation Summary

See `IMPLEMENTATION_SUMMARY.md` for complete technical details.
