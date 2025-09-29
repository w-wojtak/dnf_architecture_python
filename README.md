
This repo hosts Python code for DNF (Dynamin Neural Field) architectures from the article:

Wojtak, W., Ferreira, F., Louro, L., Bicho, E., & Erlhagen, W. (2023). Adaptive timing in a dynamic field architecture for natural human–robot interactions. Cognitive Systems Research, 82, 101148.
https://www.sciencedirect.com/science/article/pii/S1389041723000761





Run `learning_demo.py` to see an example of sequence learning in a neural field. In this demo, a sequence of five Gaussian inputs is applied to the field. Each input triggers a bump of activity that persists after the input is removed, effectively serving as a memory trace.

Due to local threshold accommodation at the bump locations, the bumps grow in amplitude over time. This leads to the formation of an activation gradient, where the largest bump corresponds to the first input, followed by progressively smaller bumps for later inputs (see Figure below).

<!-- ![ Activation gradient in a neural field. Earlier inputs produce higher bumps due to local threshold adaptation.](images/learning_example.png) -->

<img src="images/learning_example.png" alt="Activation gradient in a neural field. Earlier inputs produce higher bumps due to local threshold adaptation." width="400"/>


dnf architecture- basic

implements temporal adaptation


dnf architecture- extended



## Running Scenarios

Each experiment scenario is a script in the `scenarios/` folder (e.g., `learning_basic.py`, `recall_basic.py`).  
These scripts import shared code from `src/`, so you need to run them **from the project root**.
 

From the project root (`dnf_architecture_python/`), run:

```bash
python -m scenarios.learning_basic
python -m scenarios.recall_basic
```

### Running Combined Experiments

Some scripts orchestrate multiple scenario runs, for example:

`main_experiment.py` → runs learning_basic.py once, then recall_basic.py multiple times.

`main_experiment_extended.py` → runs learning_extended.py once, then recall_extended.py multiple times.


Run them from the project root:
```
python -m scenarios.main_experiment
python -m scenarios.main_experiment_extended
```