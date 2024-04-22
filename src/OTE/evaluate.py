# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_model
from .train_functions import test_step


# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name: str) -> float:
    """
    This function is the main program for evaluation a model under the name of 'best_model.pt'
    """

    # TODO
    # load data
    test_data: DataLoader
    _, _, test_data = load_model(name)

    # define model
    model: RecursiveScriptModule = load_model(f"{name}").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)

    return accuracy


if __name__ == "__main__":
    print(main("best_model"))
