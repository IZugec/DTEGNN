import yaml
import sys
import logging
from pathlib import Path
from types import SimpleNamespace
import torch
from torch_geometric.loader import DataLoader

from dtegnn.data.CreateDataset import DatasetCreator as CD
from dtegnn.model.nn.networks import EGNN
from dtegnn.utils.Loss import ScaledLoss
from dtegnn.utils.PrepareTrainer import get_trainer

def setup_logging(log_path: Path, log_level: int = logging.INFO) -> None:
    """
    Sets up logging configuration with both file and console handlers.
    
    Args:
        log_path: Path where log file will be stored
        log_level: Logging level (default: INFO)
    """
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_path / 'training.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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

def setup_data_loaders(args: SimpleNamespace) -> tuple[DataLoader, DataLoader]:
    """
    Sets up training and validation data loaders.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing datasets...")
    dataset_creator = CD(
            N_traj = args.N_traj,
            N_train=args.N_train,
            N_train_max=args.N_train_max,
            N_val=args.N_val,
            Num_train=args.Num_train,
            Num_val=args.Num_val,
            rmax=args.cutoff,
            neigh_factor = args.neigh_factor,
            path=args.datapath,
            potim = args.potim,
            seed=args.seed
    )

    data_train, data_val = dataset_creator.dataset()
    
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=args.batch_size_val
    )
    
    logger.info(f"Dataset initialization complete. "
                f"Training examples: {len(train_loader.dataset)}, "
                f"Validation examples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader

def create_model(args: SimpleNamespace) -> EGNN:
    """
    Creates an instance of an EGNN class based on configuration.
    
    Args:
        args: Configuration arguments
        
    Returns:
        EGNN instance
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing an instance of an EGNN model...")
    model = EGNN(
        depth=args.depth,
        hidden_features=args.emb_dim,
        node_features=len(args.atom_types_map),
        out_features=args.n_out_features,
        norm=args.norm,
        RFF_dim=args.RFF_dim,
        RFF_sigma=args.RFF_sigma
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {trainable_params:,} trainable parameters")
    
    return model

def main(config_path: Path):
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir)
    
    logger = logging.getLogger(__name__)
    
    args = read_config(config_path)
    device = torch.device("cuda" if args.cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader = setup_data_loaders(args)
    
    model = create_model(args)
    loss_fn = ScaledLoss(atom_types_map=args.atom_types_map)
    
    trainer = get_trainer(
        args=args,
        loss=loss_fn,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    logger.info("Starting training process...")
    trainer.train(device=device, args=args, n_epochs=args.n_epochs)
    logger.info("Training completed")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3.12 <path_to_dtegnn>/scripts/train.py <path_to_config>")
        sys.exit(1)
        
    config_path = Path(sys.argv[1])
    main(config_path)
