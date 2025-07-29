

Run `learning_demo.py` to see an example of sequence learning in a neural field. In this demo, a sequence of five Gaussian inputs is applied to the field. Each input triggers a bump of activity that persists after the input is removed, effectively serving as a memory trace.

Due to local threshold accommodation at the bump locations, the bumps grow in amplitude over time. This leads to the formation of an activation gradient, where the largest bump corresponds to the first input, followed by progressively smaller bumps for later inputs.