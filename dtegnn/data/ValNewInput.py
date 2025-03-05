import torch
from torch_geometric.data.batch import Batch
from ase import Atoms
from dtegnn.data.EGNNClass import EGNNClass

class VNI(torch.nn.Module):
    def __init__(self, include_si=False):
        super(VNI, self).__init__()
        self.include_si = include_si
        
    def forward(self, pos_batch, cell_vectors, num_atoms, Z , device, args ):

        pos = pos_batch.cpu().numpy()
        cell = cell_vectors.cpu().numpy()
        number_atom = num_atoms.cpu().numpy()
        atom_types = Z.cpu().numpy()

        list_of_atoms = []
        for idx, num in enumerate(num_atoms):
            ini_index = sum(number_atom[:idx])
            symbols = Z[ini_index:(ini_index + num)]
            positions = pos[ini_index:(ini_index + num)]
            cell_vec = cell[idx*3:(idx+1)*3]
            atom = Atoms(symbols=symbols, positions = positions, cell= cell_vec, pbc=[1,1,1])
            atom.wrap()
            list_of_atoms.append(atom)
        
        egnndat = EGNNClass(atoms=list_of_atoms, N = 1, cutoff = args.cutoff, essential= True)
        
        list_of_pyg_data = [egnndat[i] for i in range(len(number_atom))]

        batch = Batch()
        final_batch = batch.from_data_list(list_of_pyg_data)
        final_batch = final_batch.to(device)
        
        return final_batch
