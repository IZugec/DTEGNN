# DTEGNN

Implementation of the dynamic training method applied to an equivariant graph neural network

Correspondance to: ```zugec.ivan@gmail.com```

## Installation

Clone this repository
```commandline
git clone https://github.com/IZugec/DTEGNN.git
```
You can then build the environment with Anaconda:
```commandline
cd DTEGNN
conda env create --file envs/env-cu121.yml --force
```
This command builds an environment with CUDA 12.1 version.

## Usage

### Training

To train a DT-EGNN model you can run
```commandline
python3.12 <path_to_cloned_dir>/scripts/train.py <path_to_input_file>/config.yml
```
An example of an input configuration file can be found in dtegnn/config/example.yml.

Current implementation reads AIMD simultions with the help of ASE and assumes Trajectory object. Velocities are assumed to be in the basis of lattice vectors multiplied by the integration time of the underlying simulation.
An example of data required for a successful run can be downloaded through the following [link](https://doi.org/10.6084/m9.figshare.28498778.v1)

or directly by running
```commandline
wget https://figshare.com/ndownloader/files/52738940
```
followed by renaming the downloaded file, and unzipping it.
```commandline
mv 52738940 dataset.zip
unzip -d dataset.zip <path_to_desired_directory>
```

### Evaluation

An example showing how to deploy the model trained via DT is in our [examples directory](./examples).

