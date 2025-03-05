import yaml
import sys
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.atoms import Atoms

from dtegnn.model.nn.networks import EGNN
from dtegnn.utils.Simulate import dynamics, prepare_velocities

def read_config(filepath: Path) -> SimpleNamespace:
    """
    Reads and validates configuration from a YAML file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        SimpleNamespace object containing configuration
    """
    logger = logging.getLogger(__name__)
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {filepath}")
        return SimpleNamespace(**data)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {filepath}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file {filepath}: {str(e)}")
        raise

def main(config_path: Path, model_path: Path, output_path: Path):
    args = read_config(config_path)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = torch.load(model_path, map_location=device)

    
    indices = np.load('example_indices.npy')
    list_of_initial_structures = []
    for i in range(1,101):
        list_of_atoms_test = []
        atoms = read(args.datapath + '/traj_{}.traj'.format(i), index=':')
        vel = prepare_velocities(args.datapath + '/velall-{}'.format(i), args.potim, atoms)
        for l in range(len(atoms)):
            atoms[l].set_velocities(vel[l])
            
        for j in range(len(atoms) - args.N_eval - 1):
            list_of_atoms_test.append(atoms[j:j+args.N_eval])
            
        list_of_initial_structures.append(list_of_atoms_test[indices[i-1][1]])

    initial_structures = [structures[0] for structures in list_of_initial_structures]

    pos, vel, force, energy = dynamics(model = model,
                                       steps = int(args.N_eval*args.potim/args.dyn_potim),
                                       potim = args.dyn_potim,
                                       list_of_atoms = initial_structures ,
                                       args = args,
                                       device = device)
    
    pos_array = np.array([i.cpu().numpy() for i in pos.values()])
    vel_array = np.array([i.cpu().numpy() for i in vel.values()])
    force_array = np.array([i.cpu().numpy() for i in force.values()])
    energy_array = np.array([i.cpu().numpy() for i in energy.values()])

    np.save(output_path / 'positions.npy', pos_array)
    np.save(output_path / 'velocities.npy', vel_array)
    np.save(output_path / 'forces.npy', force_array)
    np.save(output_path / 'energies.npy', energy_array)
    
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3.12 <path_to_dtegnn>/examples/evaluate.py <path_to_config> <path_to_saved_model> <output_directory>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    main(config_path, model_path, output_path)
