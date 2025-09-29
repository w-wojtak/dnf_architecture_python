import subprocess
import sys
import os

python_exe = sys.executable
project_root = os.path.dirname(
    os.path.dirname(__file__))  # go up from scenarios/

print("=== Running learning_extended.py ===")
subprocess.run(
    [python_exe, "-m", "scenarios.learning_extended"],
    check=True,
    cwd=project_root
)

for trial_number in [1, 2, 3, 4]:
    print(f"=== Running recall_extended.py (trial {trial_number}) ===")
    subprocess.run(
        [python_exe, "-m", "scenarios.recall_extended", str(trial_number)],
        check=True,
        cwd=project_root
    )
