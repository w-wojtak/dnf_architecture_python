import subprocess
import sys
import os

python_exe = sys.executable
script_dir = os.path.dirname(__file__)

print("=== Running learning_extended.py ===")
subprocess.run([python_exe, "learning_extended.py"],
               check=True, cwd=script_dir)

for trial_number in [1, 2, 3, 4]:
    print(f"=== Running recall_extended.py (trial {trial_number}) ===")
    subprocess.run(
        [python_exe, "recall_extended.py", str(trial_number)],
        check=True,
        cwd=script_dir
    )
