import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

class FCG(torch.nn.Module):
    def __init__(self, include_si=False): 
        super(FCG, self).__init__()
        self.include_si = include_si

    def forward(self, pos_batch, cell_vectors, edge_index, cell_offset, x, batch, ptr  ):

        num_nodes = ptr[-1].item()
        batch_size = len(ptr)
        
        data_list = []
        for i in range(batch_size - 1):
            start_idx = ptr[i].item()
            end_idx = ptr[i + 1].item()
            
            tmp_data = Data()

            tmp_data.pos = pos_batch[start_idx:end_idx]
            tmp_data.x = x[start_idx:end_idx]
            tmp_data.unit_cell = cell_vectors[3*i : 3*(i+1)]
            
            edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            tmp_data.edge_index = edge_index[:, edge_mask] - start_idx
            tmp_data.cell_offset = cell_offset[edge_mask,:]

            data_list.append(tmp_data)
            
            
   
        batch = Batch()
        final_batch = batch.from_data_list(data_list)
        
        return final_batch
