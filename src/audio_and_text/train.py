# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final
import signal
import time

# own modules
from src.utils import (
    set_seed,
    save_model,
    interruption_save_handler_wrapper,
    parameters_to_double,
)
from models import Audio_Text_MSA_Model
from .train_functions import train_step, val_step

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
    dropout: float = ...
    C: int = 256
    lrn_mode: str = "full"
    lambd: float = 0.3

    # Scheduler
    weight_decay = ...
    max_iter = ...
    lr_min = ...
    milestones: list[int] = [15, 30, 60]
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
    name = f"Audio_and_Text_Model_time_{time.time()}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # MODEL
    audio_inputs, text_inputs, _, _ = next(iter(train_data)) # [batch, f, t, c], ?, _, _
    model: torch.nn.Module = Audio_Text_MSA_Model(
        audio_inputs.shape[3],
        audio_inputs.shape[1] * audio_inputs.shape[2],
        ...,
        10,
        C=C,
        lrn_mode=lrn_mode,
        lambd=lambd,
        dropout=dropout
    ).to(device)
    # Set parameters to double
    parameters_to_double(model)

    # LOSS
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), 
        ..., 
        weight_decay=weight_decay
    )

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    # EARLY STOPPING
    # register handler for ¡manual! EARLY STOPPING
    interruption_handler = interruption_save_handler_wrapper(model, name)
    signal.signal(signal.SIGINT, interruption_handler)

    # ------------TRAINING------------
    for epoch in tqdm(range(epochs)):
        # call train step
        train_loss, train_accuracy = train_step(
            model,
            train_data,
            loss,
            optimizer,
            writer,
            epoch,
            device
        )

        # call val step
        val_loss, val_accuracy = val_step(
            model,
            val_data,
            loss,
            writer,
            epoch,
            device
        )

        print(
            f"Train and Val. accuracy in epoch {epoch}, lr {scheduler.get_lr()}:",
            (round(train_accuracy, 4), round(val_accuracy, 4)),
        )

        # Compute scheduler step
        scheduler.step()

    # -------SAVING---------
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()