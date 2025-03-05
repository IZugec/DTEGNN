from pathlib import Path
import logging
import numpy as np
from typing import List, Tuple, Optional
from ase.io import read
from ase.atoms import Atoms
import random
from dtegnn.data.EGNNClass import EGNNClass

logger = logging.getLogger(__name__)

class DatasetCreator:
    """Creates and manages molecular dynamics datasets for training and validation"""

    """
        Initialize dataset creator with configuration parameters.

        Args:
            N_train: Number of steps to use in training sequences
            N_train_max: Maximum number of steps available in training sequences
            N_val: Number of steps to use in validation sequences
            Num_train: Number of training examples to sample
            Num_val: Number of validation examples to sample
            rmax: Cutoff radius for neighbor calculations
            path: Path to trajectory data
            seed: Random seed for reproducibility
        """

    
    def __init__(self, N_traj: int, N_train: int, N_train_max: int, N_val: int, 
                 Num_train: int, Num_val: int, rmax: float, neigh_factor: int, 
                 path: str, potim: float, seed: Optional[int] = None ):
        
        self.N_traj = N_traj
        self.N_train = N_train
        self.N_train_max = N_train_max
        self.N_val = N_val
        self.Num_train = Num_train
        self.Num_val = Num_val
        self.rmax = rmax
        self.neigh_factor = neigh_factor
        self.data_path = Path(path)
        self.potim = potim  
        self.seed = seed
        
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
            
        logger.info(f"Initialized DatasetCreator with {N_train} training steps and {N_val} validation steps")
    
    def dataset(self) -> Tuple[List, List]:
        """
        Creates training and validation datasets from molecular dynamics trajectories.
        
        Returns:
            Tuple of (training_dataset, validation_dataset)
        """
        logger.info("Starting dataset creation process...")
        
        try:
            # Collect trajectory data
            train_atoms, val_atoms = self._collect_trajectories()
            
            logger.info(f"Collected {len(train_atoms)} potential training and "
                       f"{len(val_atoms)} potential validation sequences")
            
            # Sample datasets
            train_set, val_set = self._sample_datasets(train_atoms, val_atoms)
            
            # Create EGNN datasets
            train_data = [
                EGNNClass(
                    atoms=atoms,
                    N=self.N_train,
                    cutoff=self.rmax,
                    potim=self.potim,
                    neigh_factor=self.neigh_factor
                )[0] for atoms in train_set
            ]
            
            val_data = [
                EGNNClass(
                    atoms=atoms,
                    N=self.N_val,
                    cutoff=self.rmax,
                    potim=self.potim,
                    validation=True
                )[0] for atoms in val_set
            ]
            
            logger.info(f"Successfully created {len(train_data)} training and "
                       f"{len(val_data)} validation examples")
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
    
    def _collect_trajectories(self) -> Tuple[List, List]:
        """Collects and processes trajectory data"""
        train_atoms = []
        val_atoms = []
        
        for i in range(1, self.N_traj):
            if i % 10 == 0:
                logger.debug(f"Processing trajectory {i}")
                
            try:
                # Load trajectory and velocity data
                traj_path = self.data_path / f'traj_{i}.traj'
                vel_path = self.data_path / f'velall-{i}'
                
                atoms = read(traj_path, index=':')
                velocities = self._prepare_velocities(vel_path, atoms)
                
                # Set velocities
                for idx, atom in enumerate(atoms):
                    atom.set_velocities(velocities[idx])
                
                # Create sequences
                for j in range(len(atoms) - self.N_train_max - 1):
                    train_atoms.append(atoms[j:j + self.N_train])
                for j in range(len(atoms) - self.N_val - 1):
                    val_atoms.append(atoms[j:j + self.N_val])
                    
            except Exception as e:
                logger.warning(f"Error processing trajectory {i}: {str(e)}")
                continue
                
        return train_atoms, val_atoms
    
    def _prepare_velocities(self, vel_path: Path, atoms: List[Atoms]) -> np.ndarray:
        """Process velocity data from file. Units of atomic velocities are assumed to be in the format given by the VASP output file OUTCAR.
        """
        with open(vel_path, 'r') as f:
            lines = f.readlines()
        
        cell_matrix = atoms[0].get_cell()
        n_atoms = len(atoms[0].get_forces())
        
        velocities = []
        for line in lines:
            div_by_potim = [float(val)/self.potim for val in line.split()]
            cart_coord = np.array(div_by_potim) @ cell_matrix  # Now in \AA/fs
            velocities.append(cart_coord)
        
        return np.array(velocities).reshape(len(atoms), n_atoms, 3)
    
    def _sample_datasets(self, train_atoms: List, val_atoms: List) -> Tuple[List, List]:
        """Samples training and validation sets"""
        if self.seed is not None:
            random.seed(self.seed)
            
        # Load or generate indices
        if Path('indices_train.npy').exists():
            logger.info("Loading existing dataset indices")
            train_indices = np.load('indices_train.npy')
            val_indices = np.load('indices_val.npy')
        else:
            logger.info("Generating new dataset indices")
            train_indices = self._generate_indices(len(train_atoms), self.Num_train)
            val_indices = self._generate_indices(len(val_atoms), self.Num_val)
            np.save('indices_train.npy', train_indices)
            np.save('indices_val.npy', val_indices)
            
        return (
            [train_atoms[i] for i in train_indices],
            [val_atoms[i] for i in val_indices]
        )
    
    def _generate_indices(self, length: int, num_indices: int) -> List[int]:
        """Generate random indices for sampling"""
        if num_indices > length:
            raise ValueError(f"Requested {num_indices} samples but only {length} available")
        return random.sample(range(length), num_indices)
