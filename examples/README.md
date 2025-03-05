# Molecular dynamics example

This example shows how to use a trained model in order to perform molecular dynamics.

## Running the example

To run this example you can execute

```commandline
python3.12 <path_to_DTEGNN>/examples/evaluate.py <path_to_config> <path_to_saved_model> <output_directory>"
```

Script ```evaluate.py``` takes three arguments, namely, a path to the input configuration file, path to the saved model and an output path denoting where to save atomic positions, velocities, energies and forces.

It uses an array of indices ```example_indices.npy``` to select atomic structures which will be used as an initial structure of a molecular dynamics.

Output positions, velocities and forces arrays have a shape of ```(N_eval, num_initial_structures * num_atoms, 3)``` while energies have a shape of ```(N_eval, num_initial_structures, 1) ``` 

## ASE

In the future, DT-EGNN will be supported as an ASE calculator. However, current in-house approach offers the advantage of performing molecular dynamics simulations in an accelerated fashion using GPU.
