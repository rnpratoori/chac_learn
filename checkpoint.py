import torch

import wandb

def save_checkpoint(epoch, model, optimizer, scheduler, epoch_losses, epoch_numbers, output_dir, filename="ch_learn_model.pth"):
    """Saves the training state to a checkpoint file."""
    checkpoint_path = output_dir / filename
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses,
        'epoch_numbers': epoch_numbers,
    }
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(state, checkpoint_path)
    
    # Create a wandb Artifact if wandb is active
    if wandb.run:
        artifact = wandb.Artifact(name="ch_learn_model", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        print(f"Saved checkpoint artifact at epoch {epoch + 1}")
    else:
        print(f"Saved checkpoint locally at epoch {epoch + 1}")

def load_checkpoint(model, optimizer, scheduler, device, output_dir, filename="ch_learn_model.pth"):
    """Load model, optimizer, and scheduler from checkpoint."""
    checkpoint_path = output_dir / filename
    
    if not checkpoint_path.exists():
        return 0, [], []
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, [], []
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    epoch_losses = checkpoint.get('epoch_losses', [])
    epoch_numbers = checkpoint.get('epoch_numbers', [])
    
    if epoch_losses:
        print(f"Checkpoint loaded. Last completed epoch: {start_epoch-1}. Loss={epoch_losses[-1]:.6e}. Resuming at epoch {start_epoch}")
    
    return start_epoch, epoch_losses, epoch_numbers