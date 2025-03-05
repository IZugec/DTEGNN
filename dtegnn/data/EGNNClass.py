#!/usr/bin/env python
# coding: utf-8
import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class EGNNClass(Dataset):
    def __init__(self, 
        N: int, atoms: list,
        cutoff: float,
        unw : list = None,
        fetch_pos: bool = False,
        fetch_vel:bool = False,
        fetch_w: bool = False,
        potim: float = 0.5,
        max_neigh: int = 4000,
        neigh_factor: int = 5,
        essential: bool = False,
        validation: bool = False,
        ):
        
        super().__init__()
        self.potim = potim
        self.atoms = atoms
        self.cutoff = cutoff
        self.N = N
        self.fetch_pos = fetch_pos
        self.fetch_vel = fetch_vel
        self.fetch_w = fetch_w
        self.unw = unw
        self.max_neigh = max_neigh
        self.neigh_factor = neigh_factor
        self.essential = essential
        self.validation = validation
        
    def get(self, idx: int):
        properties = self.get_properties(idx)
        properties["_idx"] = np.array([idx], dtype=np.int32)
        return torchify_dict(properties)
    
    def len(self):
        return len(self.atoms) - (self.N - 1)
    
    def get_properties(self,idx: int):
        outputs={}
        
        if self.essential: # These get used during training process for subsequence lengths > 1
            edge_index, edge_distances, offsets = self._get_neighbors_pymatgen(self.atoms[idx]) 
            outputs['edge_index'] = edge_index.astype(np.int64)
            outputs['pos'] = self.atoms[idx].get_positions().astype(np.float32)
            outputs['Z'] = self.atoms[idx].numbers.astype(np.int64)
            outputs['cell_offset'] = offsets.astype(np.float32)
            outputs['unit_cell'] = self.atoms[idx].get_cell().astype(np.float32)
            outputs['x'] = atoms_to_onehot_np(self.atoms[idx])
            
        else:
            edge_index, edge_distances, offsets = self._get_neighbors_pymatgen(self.atoms[idx])
            outputs['pos'] = self.atoms[idx].get_positions().astype(np.float32)
            outputs['x'] = atoms_to_onehot_np(self.atoms[idx])
            outputs['Z'] = self.atoms[idx].numbers.astype(np.int64)
            outputs['n_atoms'] = np.array([self.atoms[idx].get_global_number_of_atoms()]).astype(np.int64)
            outputs['vel'] = self.atoms[idx].get_velocities().astype(np.float32)
            outputs['unit_cell'] = self.atoms[idx].get_cell().astype(np.float32)##promjenio iz 64
            outputs['masses'] = self.atoms[idx].get_masses().astype(np.float32)
            outputs['potim'] = np.repeat(np.array([self.potim]).astype(np.float32), outputs['masses'].size )
            outputs['edge_index'] = edge_index.astype(np.int64)
            outputs['cell_offset'] = offsets.astype(np.float32)
            count=0
            
            for i in range(idx,idx+self.N):
                if not self.validation:
                    if (count+1) % self.neigh_factor == 0:
                        edge_index, edge_distances, offsets = self._get_neighbors_pymatgen(self.atoms[idx])
                        factor = (count+1) // self.neigh_factor # every 5th step we are supplementing edge index
                        outputs['edge_index' + str(factor * self.neigh_factor)] = edge_index.astype(np.int64)
                        outputs['cell_offset' + str(factor * self.neigh_factor)] = offsets.astype(np.float32)
                    
                outputs['forces' + str(count)] = self.atoms[i].get_forces().astype(np.float32)
                outputs['energy' + str(count)] = np.array([self.atoms[i].get_potential_energy()]).astype(np.float32)
                
                if (self.fetch_pos and i > idx):
                    outputs['pos' + str(count)] = self.atoms[i].get_positions()
                
                if (self.fetch_pos and i > idx):
                    outputs['vel' + str(count)] = self.atoms[i].get_velocities()
                    
                count+=1
        return outputs        
        
    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.cutoff, numerical_tol=0, exclude_self=True
        )
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]
        
        edge_index = np.vstack((_n_index, _c_index))
        edge_distances = n_distance
        cell_offsets = _offsets
        nonzero = np.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]        
                
        return edge_index, edge_distances, cell_offsets 
        
def torchify_dict(data: dict):
    torch_properties = {}
    for pname, prop in data.items():
        if prop.dtype==np.int32:
            torch_properties[pname] = torch.IntTensor(prop)
        elif prop.dtype==np.int64:
            torch_properties[pname] = torch.LongTensor(prop)
        elif prop.dtype == np.float32:
            torch_properties[pname] = torch.FloatTensor(prop.copy())
        elif prop.dtype == np.float64:
            torch_properties[pname] = torch.DoubleTensor(prop.copy())
        else:
            raise CellDataError(
                "Invalid datatype {} for property {}!".format(type(prop), pname)
            )
    final_data = Data()

    # Assign each numpy array (now a tensor) to the Data object
    for key, value in torch_properties.items():
        setattr(final_data, key, value)
        
    return final_data

def atoms_to_onehot_np(atom):
    atomic_numbers = atom.numbers
    
    unique_atomic_numbers = np.unique(atomic_numbers)
    
    mapping = {num: i for i, num in enumerate(unique_atomic_numbers)}
    mapped_atomic_numbers = np.vectorize(mapping.get)(atomic_numbers)
    
    num_atom_types = len(unique_atomic_numbers)
    one_hot_encoded = np.eye(num_atom_types)[mapped_atomic_numbers]
    
    return one_hot_encoded.astype(np.float32)
