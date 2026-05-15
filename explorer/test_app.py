#!/usr/bin/env python3
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

# Now import the app
from app import app, load_data

print("Loading data...")
load_data()
print("Data loaded successfully!")
port = int(os.environ.get('PORT', '5000'))
debug = os.environ.get('FLASK_DEBUG') == '1'
print(f"\nStarting Flask server on http://localhost:{port}")
app.run(debug=debug, host='0.0.0.0', port=port)
