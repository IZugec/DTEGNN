import time
import sys
import os
import torch
import numpy as np
import wandb
import logging
from dtegnn.model.layers.VerletLayer import VerletC,VerletV
from dtegnn.data.NewInputClass import FCG
from dtegnn.data.ValNewInput import VNI

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
    model_path,
    model,
    loss,
    optimizer,
    train_loader,
    validation_loader,
    keep_n_checkpoints=3,
    checkpoint_interval=10,
    validation_interval=1,
    schedule = None,
    cutoff = 5.0):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints")
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.keep_n_checkpoints = keep_n_checkpoints
        self._loss = loss
        self._model = model
        self.checkpoint_interval = checkpoint_interval
        self.scheduler = schedule
        self.optimizer = optimizer
        self.cutoff = cutoff
        
        self.new_input = FCG()
        self.new_input_val = VNI(include_si=False)
        self.verlet_c = VerletC()
        self.verlet_v = VerletV()
        
        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_loss = float("inf")
            self.counter = 0
            self.store_checkpoint()
            
    
    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False
        
    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    def _optimizer_to(self, device):
        """
        Move the optimizer tensors to device before training.
        Solves restore issue:
        https://github.com/atomistic-machine-learning/schnetpack/issues/126
        https://github.com/pytorch/pytorch/issues/2830
        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "start_counter": self.counter
        }
        if self._check_is_parallel():
            state_dict["model"] = self._model.module.state_dict()
        else:
            state_dict["model"] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.best_loss = state_dict["best_loss"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._load_model_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.counter = state_dict['start_counter']
    def store_checkpoint(self):
        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(self.epoch) + ".pth.tar"
        )
        torch.save(self.state_dict, chkpt)

        chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth.tar")]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))

    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            epoch = max(
                [
                    int(f.split(".")[0].split("-")[-1])
                    for f in os.listdir(self.checkpoint_path)
                    if f.startswith("checkpoint")
                ]
            )

        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(epoch) + ".pth.tar"
        )
        self.state_dict = torch.load(chkpt)
            
    def train(self, device, args ,n_epochs=sys.maxsize):
        """
        Train the model for the given number of epochs on a specified device.
        Parameters
        ----------
        args: Configuration arguments

        device : torch.device
            device on which training takes place.

        n_epochs : int
            number of training epochs.
        Notes
        -----
        """
        self._model.to(device)
        self._optimizer_to(device)
        self._stop = False

        if args.wandb_logging:
            logger.info("Initializing Weights & Biases logging")
            wandb.init(project = args.wandb_project, config = args, name = args.wandb_name, mode = args.wandb_mode)
        
        num_atom_types = len(args.atom_types_map)
        for _ in range(n_epochs):
            epoch_start_time = time.time()
            self.epoch+=1
            train_loss = 0.0
            fin_per_step = []
            fin_per_atom = []
            eng_per_step = []
            
            self._model.train()
            num_jumps = self.determine_num_jumps(args)
            self.regulate_lr(args, num_jumps)
            num_jumps = self.determine_num_jumps(args)
            self.best_model = os.path.join(self.model_path, "best_model_{}".format(num_jumps))
            logger.info(f"Epoch {self.epoch}, Number of jumps: {num_jumps}")
            logger.info(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
            
            for train_batch in self.train_loader:
                self.optimizer.zero_grad()
                train_batch = train_batch.to(device)
                
                result = {}
                auxiliary = {}
                auxiliary['pos0'] = train_batch['pos']
                auxiliary['vel0'] = train_batch['vel']

                f_energy, f_force = self._model(train_batch, train_flag = True)

                result['energy0'] = f_energy
                result['forces0'] = f_force
                
                for jump in range(num_jumps - 1):
                    
                    factor = (jump+1)//args.neigh_factor

                    auxiliary['pos' + str(jump+1) ] = self.verlet_c(result['forces' + str(jump)] , 
                                                          auxiliary['pos' + str(jump)], 
                                                          auxiliary['vel' +str(jump)],
                                                          train_batch['potim'], 
                                                          train_batch['masses'], 
                                                          train_batch['unit_cell'])
                    if factor == 0:
                        new_input = self.new_input(auxiliary['pos' + str(jump+1)],
                                                              train_batch['unit_cell'],
                                                              train_batch['edge_index' ],
                                                              train_batch['cell_offset' ],
                                                              train_batch['x'],
                                                              train_batch['batch'],
                                                              train_batch['ptr'])
                    else:
                        new_input = self.new_input(auxiliary['pos' + str(jump+1)],
                                                              train_batch['unit_cell'],
                                                              train_batch['edge_index' + str(factor * args.neigh_factor) ],
                                                              train_batch['cell_offset' + str(factor * args.neigh_factor) ],
                                                              train_batch['x'],
                                                              train_batch['batch'],
                                                              train_batch['ptr'])
                        
                                        
                    energy, force = self._model(new_input, train_flag=True)
                    
                    result['energy' + str(jump+1)] = energy
                    result['forces' + str(jump+1)] = force
                    
                    auxiliary['vel' + str(jump + 1)] = self.verlet_v(result['forces' + str(jump + 1)],
                                                                      result['forces' + str(jump)],
                                                                      auxiliary['vel' + str(jump)],
                                                                      train_batch['potim'],
                                                                      train_batch['masses'])
                
                per_step, per_atom, eng_step = self._loss(train_batch, result, num_jumps, auxiliary)
                
                if num_jumps>1:
                    loss = sum([per_step[0] * args.first_term] + per_step[1:])

                else:
                    loss = sum(per_step) + sum(eng_step)

                fin_per_step.append([ten.cpu().detach().numpy() for ten in per_step])
                fin_per_atom.extend(per_atom)
                eng_per_step.append([ten.cpu().detach().numpy() for ten in eng_step])
                
                loss.backward()
                clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                
            fin_per_step = np.array(fin_per_step)
            summed_per_step = np.sum(fin_per_step, axis=0)/len(fin_per_step)

            eng_per_step = np.array(eng_per_step)
            summed_eng = np.sum(eng_per_step, axis=0)/len(eng_per_step)
            

            logger.info(f"Train mean ENERGY PER ATOM loss per STEP: {summed_eng}")
            logger.info(f"Train mean FORCE component loss per STEP: {summed_per_step}")

            train_means = calculate_means(fin_per_atom, num_atom_types)

            atom_type_losses = []
            for idx, atom_name in args.atom_types_map.items():
                atom_type_losses.append(f"{atom_name}: {train_means[idx]:.6f}")

            logger.info(f"Train mean FORCE component loss per ATOM TYPE: {', '.join(atom_type_losses)}")


            if self.epoch % args.check_val == 0:
                self._model.eval()
                fin_per_step_val = []
                fin_per_atom_val = []
                fin_eng_per_step_val = []
                val_loss = 0.0
                
                for val_batch in self.validation_loader:
                    val_batch = val_batch.to(device)
                    val_result = {}
                    auxiliary = {}
                    auxiliary['pos0'] = val_batch['pos']
                    auxiliary['vel0'] = val_batch['vel']
                    f_energy, f_force = self._model(val_batch, train_flag=False)
                    
                    val_result['energy0'] = f_energy.detach()
                    val_result['forces0'] = f_force.detach()
                    
                    for jump in range(args.N_val-1):
                        
                        auxiliary['pos' + str(jump+1)] = self.verlet_c(val_result['forces' + str(jump)] , 
                                                          auxiliary['pos' + str(jump)], 
                                                          auxiliary['vel' +str(jump)],
                                                          val_batch['potim'], 
                                                          val_batch['masses'], 
                                                          val_batch['unit_cell'])
                        new_input = self.new_input_val(auxiliary['pos' + str(jump+1)],
                                                       val_batch['unit_cell'],
                                                       val_batch['n_atoms'],
                                                       val_batch['Z'],
                                                       device,
                                                       args)
                        energy, force = self._model(new_input, train_flag = False)
                        
                        val_result['energy' + str(jump+1)] = energy.detach()
                        val_result['forces' + str(jump+1)] = force.detach()
                        
                        auxiliary['vel' + str(jump + 1)] = self.verlet_v(val_result['forces' + str(jump + 1)],
                                                                      val_result['forces' + str(jump)],
                                                                      auxiliary['vel' + str(jump)],
                                                                      val_batch['potim'],
                                                                      val_batch['masses'])
                            
                    per_step_val, per_atom_val, eng_step_val = self._loss(val_batch, val_result, args.N_val, auxiliary)
                    
                    val_batch_loss = sum(per_step_val)
                    val_batch_loss = val_batch_loss.data.cpu().numpy()
                    fin_per_step_val.append([ten.cpu() for ten in per_step_val])
                    fin_per_atom_val.extend(per_atom_val)
                    fin_eng_per_step_val.append([ten.cpu() for ten in eng_step_val])
                    
                    val_loss += val_batch_loss
                    
                if self.best_loss > val_loss:
                    self.best_loss = val_loss
                    torch.save(self._model, self.best_model)
                
                fin_per_step_val = np.array(fin_per_step_val)
                summed_per_step_val = np.sum(fin_per_step_val, axis=0)/len(fin_per_step_val)
                
                self.scheduler.step(val_loss) 
                
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                val_means = calculate_means(fin_per_atom_val, num_atom_types)
                logger.info(f"Validation mean FORCE component loss per STEP: {summed_per_step_val}")
                val_atom_type_losses = []
                for idx, atom_name in args.atom_types_map.items():
                    val_atom_type_losses.append(f"{atom_name}: {val_means[idx]:.6f}")

                logger.info(f"Validation mean FORCE component loss per ATOM TYPE: {', '.join(val_atom_type_losses)}")


                if args.wandb_logging:
                    log_dict = {
                        'Train Mean FC error': np.mean(summed_per_step),
                        'Val Mean FC error': np.mean(summed_per_step_val),
                        'epoch_time_seconds': epoch_duration,
                        'Learning rate': self.scheduler._last_lr[0]
                        }
                    for idx, atom_name in args.atom_types_map.items():
                        log_dict[f"{atom_name} mean train loss per step"] = train_means[idx]
                        log_dict[f"{atom_name} mean val loss per step"] = val_means[idx]

                
                    wandb.log(log_dict, step=self.epoch)
    
            else:
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                if args.wandb_logging:
                    log_dict = {
                    'Train Mean FC error': np.mean(summed_per_step),
                    'epoch_time_seconds': epoch_duration,
                    'Learning rate': self.scheduler._last_lr[0]
                    }
                    wandb.log(log_dict, step=self.epoch)
            self.store_checkpoint()

    def determine_num_jumps(self, args):
        steps = [i for i in range(1, args.N_train+1)]
        assert self.counter < len(steps), "Maximum number of steps reached. Please increase N_train in the input file."
        
        return steps[self.counter]
    
    def regulate_lr(self, args, num_jumps):
        current_lr = self.optimizer.param_groups[0]['lr']
        if num_jumps <= args.N_train:
            if current_lr <= args.lr_min:
                for g in self.optimizer.param_groups:
                    g['lr'] = args.lr_reset
                    self.scheduler = ReduceLROnPlateau(self.optimizer, factor = args.lr_decay, patience= args.patience, min_lr = args.lr_min)
                self.counter+= 1
                self.best_loss = float("inf")
        else:
            return None

def calculate_means(loss_list, num_atom_types):
    """Calculate mean losses for each atom type.
    
    Args:
        loss_list: List of losses
        num_atom_types: Number of different atom types
        
    Returns:
        List of mean losses for each atom type in order of atom_types_map keys
    """
    sums = [0.0] * num_atom_types
    counts = [0] * num_atom_types

    for i, loss in enumerate(loss_list):
        index = i % num_atom_types
        sums[index] += loss
        counts[index] += 1

    means = [sums[i] / counts[i] for i in range(num_atom_types)]
    return means
    
    
        
