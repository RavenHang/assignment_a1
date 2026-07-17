import numpy as np
import torch
import os
from typing import BinaryIO, IO

def get_batch(
    data: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str | torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = len(data) - context_length
    if max_idx <= 0:
        raise ValueError(
            f"Dataset has {len(data)} tokens, but context_length={context_length} requires at least "
            f"{context_length + 1} tokens."
        )

    if rng is None:
        starts = np.random.randint(0, max_idx, size=batch_size)
    else:
        starts = rng.integers(0, max_idx, size=batch_size)

    offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
    batch = np.asarray(data[offsets], dtype=np.int64)
    batch_tensor = torch.from_numpy(batch).to(device=device, non_blocking=True)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    
    torch.save(checkpoint_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    
    checkpoint_dict = torch.load(src, map_location="cpu")    
    model.load_state_dict(checkpoint_dict["model_state_dict"])    
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])    
    return checkpoint_dict["iteration"]
