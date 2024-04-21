# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final
import signal
import time
import warnings
warnings.filterwarnings("ignore")

# own modules
from src.utils import (
    set_seed,
    save_model,
    interruption_save_handler_wrapper,
    parameters_to_double,
)
from models import OTE_Model
from .train_functions import train_step, val_step
from src.data.SentiCap import load_data

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
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 26
    dropout: float = 0.4
    image_out_dim = 64
    text_out_dim= 64
    classification_hidden_size= 70
    attention_heads= 4 # ¡¡ (image_out_dim + text_out_dim) % attention_heads == 0 !!
    use_small_cnn: bool = True
    dim_feed_forward=1024

    # Scheduler
    weight_decay = 0.01
    gamma = 0.1
    milestones: list[int] = [15, 30, 60]
    # -----------------------

    # -------LOADING---------
    print("Loading data...")
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _, = load_data(batch_size=batch_size)
    print("DONE")

    # ------------PRE-TRAINING-----------

    # empty nohup file
    open("nohup.out", "w").close()

    # SAVING PATH AND PROGRESS TRACKER
    name = f"OTE_Model_time_{time.time()}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # MODEL
    image_inputs, text_inputs , _ = next(iter(train_data)) # [batch, in_channels, h, w], [batch, seq, embedding dim], _
    model: torch.nn.Module = OTE_Model(
        image_in_channels=image_inputs.shape[1],
        image_out_dim=image_out_dim,
        text_seq_dim=text_inputs.shape[1],
        text_embedding_dim=text_inputs.shape[2],
        text_out_dim=text_out_dim,
        classifier_hidden_size=classification_hidden_size,
        num_classes=2,
        dropout=dropout,
        num_heads=attention_heads,
        use_small_cnn=use_small_cnn,
        dim_feed_forward=dim_feed_forward
    ).to(device)
    # Set parameters to double
    parameters_to_double(model)

    # LOSS
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

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
