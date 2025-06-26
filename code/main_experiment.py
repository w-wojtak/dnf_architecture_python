import subprocess

# Step 1: Run learning phase
print("=== Running learning_two_fields.py ===")
subprocess.run(["python", "learning_two_fields.py"], check=True)

# Step 2: Run recall phase 3 times
for trial_number in [1, 2, 3]:
    print(
        f"\n=== Running recall_two_fields.py with trial_number={trial_number} ===")
    subprocess.run(["python", "recall_two_fields.py",
                   str(trial_number)], check=True)
