from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from ase.atoms import Atoms
from torch_geometric.data.batch import Batch
from dtegnn.model.layers.VerletLayer import VerletC,VerletV
from dtegnn.data.ValNewInput import VNI    
from dtegnn.data.EGNNClass import EGNNClass

def prepare_velocities(vel_path: Path, potim: float, atoms: List[Atoms]) -> np.ndarray:
        """Process velocity data from file. Units of atomic velocities are assumed to be in the format given by the VASP output file OUTCAR.
        """
        with open(vel_path, 'r') as f:
            lines = f.readlines()

        cell_matrix = atoms[0].get_cell()
        n_atoms = len(atoms[0].get_forces())

        velocities = []
        for line in lines:
            div_by_potim = [float(val)/potim for val in line.split()]
            cart_coord = np.array(div_by_potim) @ cell_matrix  # Now in \AA/fs
            velocities.append(cart_coord)

        return np.array(velocities).reshape(len(atoms), n_atoms, 3)


def dynamics(model,steps,potim,list_of_atoms,args,device):    
    """
    Performs molecular dynamics simulation on atomic systems using a trained model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model that predicts energy and forces.
    
    steps : int
        Number of simulation timesteps to perform.
    
    potim : float
        Integration time in femtoseconds.
    
    list_of_atoms : list of ase.Atoms
        List of ASE Atoms objects representing initial configurations.
    
    args : argparse.Namespace
        Command-line arguments containing simulation parameters.
        Required to have args.cutoff (radial cutoff distance).
    
    device : torch.device
        Device (CPU or GPU) for computations.
    
    Returns
    -------
    pos_dict : dict
        Dictionary of atomic positions at each timestep.
        Keys format: 'pos0', 'pos1', ..., 'pos{steps-1}'
    
    vel_dict : dict
        Dictionary of atomic velocities at each timestep.
        Keys format: 'vel0', 'vel1', ..., 'vel{steps-1}'
    
    force_dict : dict
        Dictionary of forces at each timestep.
        Keys format: 'forces0', 'forces1', ..., 'forces{steps-1}'
    
    energy_dict : dict
        Dictionary of system energies at each timestep.
        Keys format: 'energy0', 'energy1', ..., 'energy{steps-1}'
    """
    vel_dict = {}
    pos_dict = {}
    force_dict = {}
    energy_dict = {}
    new_input_inf = VNI(include_si=False)
    verlet_c = VerletC()
    verlet_v = VerletV()
    egnndat = EGNNClass(atoms=list_of_atoms, N = 1, cutoff = args.cutoff,potim=potim)
    
    list_of_pyg_data = [egnndat[i] for i in range(len(list_of_atoms))]

    empty_batch = Batch()
    full_batch = empty_batch.from_data_list(list_of_pyg_data)
    full_batch = full_batch.to(device)
    
    vel_dict['vel0'] = full_batch['vel']
    pos_dict['pos0'] = full_batch['pos']

    model.eval()
    
    f_energy, f_force = model(full_batch, train_flag=False)
    energy_dict['energy0'] = f_energy.detach()
    force_dict['forces0'] = f_force.detach()
    
    
    for step in range(steps-1):
        pos_dict['pos' + str(step+1)] = verlet_c(force_dict['forces' + str(step)] , 
                                                      pos_dict['pos' + str(step)], 
                                                      vel_dict['vel' +str(step)],
                                                      full_batch['potim'], 
                                                      full_batch['masses'], 
                                                      full_batch['unit_cell'])
        
        new_input = new_input_inf(pos_dict['pos' + str(step+1)],
                                                   full_batch['unit_cell'],
                                                   full_batch['n_atoms'],
                                                   full_batch['Z'],
                                                   device,
                                                   args,
                                                    )

        pos_dict['pos' + str(step+1)] = new_input.pos
        

        energy, force = model(new_input)
        energy_dict['energy' + str(step+1)] = energy.detach()
        force_dict['forces' + str(step+1)] = force.detach()

        vel_dict['vel' + str(step + 1)] = verlet_v(force_dict['forces' + str(step + 1)],
                                                                  force_dict['forces' + str(step)],
                                                                  vel_dict['vel' + str(step)],
                                                                  full_batch['potim'],
                                                                  full_batch['masses'])
    return pos_dict, vel_dict, force_dict, energy_dict
