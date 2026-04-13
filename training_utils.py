
import os
import argparse
import random
import numpy as np
import torch

import torch.optim as optim
import wandb
from pathlib import Path
from checkpoint import load_checkpoint



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Cahn-Hilliard learning script.')
    parser.add_argument('--epochs', type=int, default=10000, 
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, 
                        help='Learning rate for optimizer.')
    parser.add_argument('--resume-lr', type=float, default=None,
                        help='Learning rate to use when resuming from checkpoint. '
                             'If not specified, uses --learning-rate value.')
    parser.add_argument('--seed', type=int, default=12, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--no-resume', action='store_true', 
                        help='Start training from scratch, ignoring checkpoints.')
    parser.add_argument('--no-wandb', action='store_true', 
                        help='Disable Weights & Biases logging.')
    parser.add_argument('--no-scheduler', action='store_true',
                        help='DEPRECATED: Use --scheduler none instead. Disable learning rate scheduler.')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none', 'plateau'],
                        help='Learning rate scheduler type.')
    parser.add_argument('--warmup-epochs', type=int, default=100,
                        help='Number of epochs for learning rate warm-up (only for cosine scheduler).')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for ReduceLROnPlateau scheduler.')
    parser.add_argument('--factor', type=float, default=0.8,
                        help='Factor for ReduceLROnPlateau scheduler.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results.')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling mode (reduces epochs to 2).')
    parser.add_argument('--cpu', action='store_true',
                        help='Force usage of CPU for PyTorch even if CUDA is available.')
    # Mesh parameters (2D — always a square RectangleMesh)
    parser.add_argument('--N', type=int, default=100,
                        help='Number of mesh cells per axis (RectangleMesh(N, N, L, L)).')
    parser.add_argument('--L', type=float, default=1.0,
                        help='Domain length per axis.')
    parser.add_argument('--eta-loss-weight', type=float, default=1.0,
                        help='Weight for the crystallinity (eta) loss term relative to concentration (c) loss.')
    parser.add_argument('--integrability-weight', type=float, default=1.0,
                        help='Weight for the integrability constraint (mixed partials equality) in the loss.')
    parser.add_argument('--integrability-start-epoch', type=int, default=0,
                        help='Epoch at which to start adding the integrability regularizer.')
    # Physics parameters
    parser.add_argument('--chi', type=float, default=1.0,
                        help='Flory-Huggins interaction parameter chi.')
    parser.add_argument('--chi-ac', type=float, default=1.0,
                        help='Coupling parameter chi_ac.')
    parser.add_argument('--N1', type=float, default=3.0,
                        help='Degree of polymerization N1.')
    parser.add_argument('--N2', type=float, default=3.0,
                        help='Degree of polymerization N2.')
    parser.add_argument('--Weta', type=float, default=1.0,
                        help='Barrier height Weta.')
    parser.add_argument('--z', type=float, default=0.5,
                        help='Parameter z.')
    parser.add_argument('--z0', type=float, default=1.0,
                        help='Parameter z0.')
    # Simulation parameters
    parser.add_argument('--T', type=float, default=1e-1,
                        help='Total simulation time.')
    parser.add_argument('--dt', type=float, default=1e-4,
                        help='Time step size.')
    parser.add_argument('--M', type=float, default=1.0,
                        help='Mobility parameter.')
    parser.add_argument('--lmbda-eta', type=float, default=5e-2,
                        help='Interface width parameter for eta.')
    parser.add_argument('--L-kinetic', type=float, default=-100.0,
                        help='Kinetic coefficient for Allen-Cahn.')
    parser.add_argument('--data-index', type=int, default=1,
                        help='Index of the reference data directory (e.g., 1 for ch_ac_1).')
    return parser.parse_args()

def setup_device(args):
    """Setup PyTorch device (CUDA or CPU)."""
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using PyTorch device: {device}")
    return device

def setup_output_dir(args):
    """Setup and return the output directory path."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(os.getenv("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def initialize_training(args, model, device, output_dir, checkpoint_filename="ch_learn_model.pth"):
    """Initialize optimizer, scheduler, and wandb."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_default_dtype(torch.float64)
    
    # Ensure model is on correct device and dtype
    model = model.to(device)
    model.double()
    
    # Create optimizer and conditionally create scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    
    # Handle deprecated --no-scheduler flag
    if args.no_scheduler:
        print("Warning: --no-scheduler is deprecated. Use --scheduler none instead.")
        args.scheduler = 'none'

    if args.scheduler == 'cosine':
        print(f"Using cosine annealing scheduler with {args.warmup_epochs} warm-up epochs.")
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-5
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    elif args.scheduler == 'plateau':
        print(f"Using ReduceLROnPlateau scheduler with patience {args.patience} and factor {args.factor}. Warmup handled in training loop.")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.factor,
            patience=args.patience
        )
    elif args.scheduler == 'none':
        print("Learning rate scheduler is disabled.")
    else: # This else block will now be for 'plateau', which is removed.
        # This part should ideally not be reached if choices are restricted in argparse.
        # For safety, we can print a message.
        print(f"Scheduler '{args.scheduler}' is not supported. Training without a scheduler.")

    # Load checkpoint if available
    start_epoch = 0
    epoch_losses = []
    epoch_numbers = []
    resumed = False
    
    if not args.no_resume:
        start_epoch, epoch_losses, epoch_numbers = load_checkpoint(
            model, optimizer, scheduler, device, output_dir, filename=checkpoint_filename
        )
        if start_epoch > 0:
            resumed = True

    # Determine learning rate to use
    if resumed and args.resume_lr is not None:
        lr = args.resume_lr
        print(f"Overriding learning rate to: {lr} (keeping optimizer momentum state)")
        for g in optimizer.param_groups:
            g['lr'] = lr
    else:
        lr = optimizer.param_groups[0]['lr']
        if resumed:
            print(f"Resumed with learning rate: {lr}")
    
    # Initialize wandb
    if not args.no_wandb:
        config = {
            "learning_rate": lr,
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
            "resumed": resumed,
            "scheduler": args.scheduler,
        }
        if args.scheduler == 'cosine':
            config["warmup_epochs"] = args.warmup_epochs
        elif args.scheduler == 'plateau':
            config["warmup_epochs"] = args.warmup_epochs
            config["patience"] = args.patience
            config["factor"] = args.factor
        
        if resumed and args.resume_lr is not None:
            config["resume_lr"] = args.resume_lr
        config["N"] = args.N
        config["L"] = args.L
        config["eta_loss_weight"] = args.eta_loss_weight
        config["integrability_weight"] = args.integrability_weight
        config["integrability_start_epoch"] = args.integrability_start_epoch
        wandb.init(project="ch_learn", config=config, resume="allow")
    
    return model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers
