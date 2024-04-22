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
from src.data.IEMOCAP import load_data

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    # ----HYPERPARAMETERS----
    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 64
    dropout: float = 0.4
    C: int = 256
    lrn_mode: str = "full"
    lambd: float = 0.3
    time_dim: int = 750
    out_text_dim = 750

    # Scheduler
    weight_decay = 0.01
    gamma = 0.1
    milestones: list[int] = [15, 30, 40]
    # -----------------------

    # -------LOADING---------
    print("Loading data...")
    train_data: DataLoader
    val_data: DataLoader
    (
        train_data,
        val_data,
        _,
    ) = load_data(batch_size=batch_size, time_dim=time_dim)
    print("DONE")

    # ------------PRE-TRAINING-----------

    # empty nohup file
    open("nohup.out", "w").close()

    # SAVING PATH AND PROGRESS TRACKER
    name = f"Audio_and_Text_Model_time_{time.time()}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # MODEL
    audio_inputs, text_inputs, _ = next(
        iter(train_data)
    )  # [batch, f, t, c], [batch, seq, embedding dim], _
    model: torch.nn.Module = Audio_Text_MSA_Model(
        f_t_c=audio_inputs.shape[1:],
        num_classes=10,
        seq_dim=text_inputs.shape[1],
        embedding_dim=text_inputs.shape[2],
        out_text_dim=out_text_dim,
        C=C,
        lrn_mode=lrn_mode,
        lambd=lambd,
        dropout=dropout,
    ).to(device)
    # Set parameters to double
    parameters_to_double(model)

    # LOSS
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    # EARLY STOPPING
    # register handler for ¡manual! EARLY STOPPING
    interruption_handler = interruption_save_handler_wrapper(model, name)
    signal.signal(signal.SIGINT, interruption_handler)

    # ------------TRAINING------------
    for epoch in tqdm(range(epochs)):
        # call train step
        train_loss, train_accuracy = train_step(
            model, train_data, loss, optimizer, writer, epoch, device
        )

        # call val step
        val_loss, val_accuracy = val_step(model, val_data, loss, writer, epoch, device)

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
