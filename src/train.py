# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final
import signal

# own modules
from src.utils import (
    set_seed,
    save_model,
    interruption_save_handler_wrapper,
    parameters_to_double,
)

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    # ----HYPERPARAMETERS----
    epochs: int = ...
    lr: float = ...
    batch_size: int = ...
    past_days: int = ...
    hidden_size: int = ...
    num_layers: int = ...
    dropout: float = ...
    bidirectional = ...

    # Scheduler
    weight_decay = ...
    max_iter = ...
    lr_min = ...
    # -----------------------

    # -------LOADING---------
    print("Loading data...")
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _, mean, std = ...
    print("DONE")

    # ------------PRE-TRAINING-----------

    # empty nohup file
    open("nohup.out", "w").close()

    # SAVING PATH AND PROGRESS TRACKER
    name = ...
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # MODEL
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = ...().to(device)
    # Set parameters to double
    parameters_to_double(model)

    # LOSS
    loss: torch.nn.Module = ...

    # OPTIMIZER
    optimizer: torch.optim.Optimizer = ...

    # SCHEDULER
    scheduler = ...

    # EARLY STOPPING
    # register handler for ¡manual! EARLY STOPPING
    interruption_handler = interruption_save_handler_wrapper(model, name)
    signal.signal(signal.SIGINT, interruption_handler)

    # ------------TRAINING------------
    for epoch in tqdm(range(epochs)):
        # call train step
        ...

        # call val step
        ...

        print(
            f"Train and Val. mae in epoch {epoch}, lr {scheduler.get_lr()}:",
            (round(..., 4), round(..., 4)),
        )

        # Compute scheduler step
        scheduler.step()

    # -------SAVING---------
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
