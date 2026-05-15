# BixBench Explorer

A Flask web application for exploring BixBench bioinformatics benchmark results.

## Features

- **Dashboard**: Aggregate metrics and visualizations comparing GPT-4o and Claude 3.5 Sonnet
- **Question Explorer**: Browse and filter 205 questions with model predictions
- **Capsule Explorer**: View capsule-level statistics and lazy-load data from HuggingFace
- **Model Comparison**: Analyze agreement and disagreement between models

## Setup Instructions

### Using UV (Recommended - Fast!)

UV is the fastest way to set up the environment:

```bash
cd /Users/mmingay/testing/biobench/explorer

# Create venv with ARM64 Python
uv venv --python /opt/homebrew/bin/python3

# Activate the venv
source .venv/bin/activate

# Install dependencies (UV is super fast!)
uv pip install flask pandas huggingface_hub

# Run the app
python app.py
```

The app will be available at: **http://localhost:5001**

### Option 1: Create a new ARM64 environment

```bash
# Create a fresh virtual environment
cd /Users/mmingay/testing/biobench/explorer
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install flask pandas huggingface_hub

# Run the app
python app.py
```

### Option 2: Use conda with ARM64

```bash
# Create a new conda environment with ARM64 support
CONDA_SUBDIR=osx-arm64 conda create -n biobench python=3.10

# Activate it
conda activate biobench

# Install dependencies
conda install flask pandas
pip install huggingface_hub

# Run the app
python app.py
```

### Option 3: Use system Python (if available)

```bash
# Check if system Python is ARM64
/usr/bin/python3 --version

# Install dependencies for user
/usr/bin/python3 -m pip install --user flask pandas huggingface_hub

# Run the app
/usr/bin/python3 app.py
```

## Running the App

Once dependencies are installed correctly:

```bash
cd /Users/mmingay/testing/biobench/explorer
python app.py
```

The app will be available at: **http://localhost:5001**

## Deploying on Render Free

Create a Render Web Service pointed at this repository:

- **Root directory:** `explorer`
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn wsgi:application --bind 0.0.0.0:$PORT`

The free web service filesystem is ephemeral, which is fine for this app's capsule cache. Downloaded capsule files are cached under `explorer/cache/` while the instance is alive and may disappear after restarts, redeploys, or spin-downs.

Optional environment variables:

- `MAX_CAPSULE_ZIP_MB` defaults to `250`
- `MAX_CAPSULE_EXTRACTED_MB` defaults to `750`
- `MIN_FREE_DISK_MB` defaults to `200`
- `FLASK_DEBUG=1` enables debug mode only for local `python app.py` runs
- `PORT` controls the local `python app.py` port and defaults to `5001`

## Routes

- `/` - Dashboard with aggregate metrics
- `/questions` - Browse all questions with filters
- `/questions/<uuid>` - View detailed question information
- `/capsules` - List all capsules with statistics
- `/capsules/<capsule_id>` - View capsule details and load data files
- `/compare` - Compare model predictions

## Data Sources

The app automatically loads (paths relative to `explorer/`):
- v1.5 results: `BixBench/bixbench-v1.5_results/zero_shot_baselines/*.csv`
- Original results: `BixBench/bixbench_results/baseline_eval_data/*.csv`
- Capsule data: Downloaded on-demand from HuggingFace (`futurehouse/bixbench`)

## File Structure

```
explorer/
├── BixBench/           # Baseline CSVs/JSON (v1.5 + original results)
├── app.py              # Flask application
├── templates/
│   ├── base.html       # Base template
│   ├── dashboard.html  # Dashboard page
│   ├── questions.html  # Questions list
│   ├── question.html   # Question detail
│   ├── capsules.html   # Capsules list
│   ├── capsule.html    # Capsule detail
│   └── compare.html    # Model comparison
├── cache/              # Capsule data cache (created on-demand)
└── requirements.txt    # Python dependencies
```

## Troubleshooting

If you see numpy architecture errors:
1. Make sure you're using an ARM64-compatible Python environment
2. Completely remove and reinstall numpy/pandas
3. Use a virtual environment to avoid conflicts with existing installations

## Features

### Dashboard
- Bar charts showing accuracy by configuration
- Coverage vs Precision comparison
- Random baseline indicators (25% for MCQ no-refusal, 20% for MCQ with-refusal)

### Question Explorer
- Filter by capsule, evaluation mode, model, and correctness
- Pagination for easy navigation
- Color-coded correctness indicators

### Capsule Detail
- Per-capsule accuracy statistics
- Lazy-load data files from HuggingFace
- File preview functionality

### Model Comparison
- Categorizes questions by agreement:
  - Both correct
  - Both wrong
  - GPT-4o only correct
  - Claude only correct
- Highlights model-specific strengths and weaknesses
