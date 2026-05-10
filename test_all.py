
import subprocess
import sys

modules = [
    ("src/grid_model.py", "Grid Model"),
    ("src/layout_validator.py", "Layout Validator"),
    ("src/fleet_selector.py", "Fleet Selector"),
    ("src/astar_planner.py", "A* Planner"),
    ("src/ml_pipeline.py", "ML Pipeline"),
    ("src/load_datasets.py", "Dataset Loader"),
]

print("="*60)
print("TESTING ALL AERONET LITE MODULES")
print("="*60)

for module, name in modules:
    print(f"\n\n--- Testing {name} ---")
    print("-"*40)
    result = subprocess.run([sys.executable, module], capture_output=False)
    
print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)