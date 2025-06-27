import subprocess
import sys
import os

python_exe = sys.executable
script_dir = os.path.dirname(__file__)

print("=== Running learning_two_fields.py ===")
subprocess.run([python_exe, "learning_two_fields.py"],
               check=True, cwd=script_dir)

for trial_number in [1, 2, 3]:
    print(f"=== Running recall_two_fields.py (trial {trial_number}) ===")
    subprocess.run(
        [python_exe, "recall_two_fields.py", str(trial_number)],
        check=True,
        cwd=script_dir
    )
