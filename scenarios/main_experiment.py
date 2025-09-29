# main_experiment.py
import subprocess
import sys
import os

python_exe = sys.executable
project_root = os.path.dirname(
    os.path.dirname(__file__))  # go up from scenarios/

print("=== Running learning_basic.py ===")
subprocess.run([python_exe, "-m", "scenarios.learning_basic"],
               check=True, cwd=project_root)

for trial_number in [1, 2, 3, 4]:
    print(f"=== Running recall_basic.py (trial {trial_number}) ===")
    subprocess.run([python_exe, "-m", "scenarios.recall_basic", str(trial_number)],
                   check=True,
                   cwd=project_root)
