import torch
import torch.nn as nn

class ScaledLoss(nn.Module):
    def __init__(self, atom_types_map=None):
        super(ScaledLoss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.atom_types_map = atom_types_map

    def _get_atom_masks(self, atom_types):
        """
        Args:
            atom_types: One-hot encoded atom types tensor [num_atoms, num_types]
        
        Returns:
            List of boolean masks, ordered according to atom_types_map
        """
        num_atom_types = atom_types.size(1)
        masks = []
        
        # Create masks in the order of one-hot encoding positions
        for type_idx in range(num_atom_types):
            mask = atom_types[:, type_idx] == 1
            masks.append(mask)
            
        return masks

    def forward(self, true, predict, num_jumps, auxiliary):
        
        batch_size = true['energy0'].size(0)
        per_step = []
        per_atom = []
        eng_step = []
        
        atom_masks = self._get_atom_masks(true['x'])
        num_atoms = len(true['forces0']) / batch_size

        for i in range(num_jumps):
            energy_loss = self.mae_loss(predict['energy' + str(i)].squeeze(1)/num_atoms, true['energy' + str(i)]/num_atoms)
            per_step_sum = 0.0
            for mask in atom_masks:
                atom_predicted_forces = predict['forces' + str(i)][mask]
                atom_dft_forces = true['forces' + str(i)][mask]
                atom_loss = self.mae_loss(atom_predicted_forces, atom_dft_forces)
                per_atom.append(atom_loss.item())
                per_step_sum += atom_loss
                
            per_step.append(per_step_sum)
            eng_step.append(energy_loss)
            
        return per_step, per_atom, eng_step
