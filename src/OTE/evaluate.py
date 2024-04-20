# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_model

# static variables
DATA_PATH: Final[str] = "data"

# set device
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
set_seed(42)


def main(name: str) -> float:
    """
    This function is the main program.
    """

    # TODO
    # load data
    test_data: DataLoader
    _, _, test_data, mean, std = ...

    # define model
    model: RecursiveScriptModule = load_model(f"{name}").to(device)

    # call test step and evaluate accuracy
    accuracy: float = ...

    return accuracy


if __name__ == "__main__":
    print(main("best_model"))