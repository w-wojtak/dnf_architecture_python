import subprocess
import sys
import os

python_exe = sys.executable
script_dir = os.path.dirname(__file__)

print("=== Running learning_basic.py ===")
subprocess.run([python_exe, "learning_basic.py"],
               check=True, cwd=script_dir)

for trial_number in [1, 2, 3, 4]:
    print(f"=== Running recall_basic.py (trial {trial_number}) ===")
    subprocess.run(
        [python_exe, "recall_basic.py", str(trial_number)],
        check=True,
        cwd=script_dir
    )
