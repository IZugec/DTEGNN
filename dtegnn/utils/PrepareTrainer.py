import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dtegnn.training.Trainer import Trainer


def get_trainer(args, model, loss, train_loader, val_loader):
   
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, factor = args.lr_decay, patience= args.patience, min_lr = args.lr_min)
    
    trainer = Trainer(
        args.modelpath,
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        schedule=scheduler,
        cutoff = args.cutoff
    )
    return trainer
