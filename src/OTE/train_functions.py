# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import compute_accuracy

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.

    Return:
        tuple[float, float]: the loss and accuracy in train step
    """

    # TODO

    # Define metric lists
    # Loss
    losses: list[float] = []
    # Accuracy
    accuracies: list[float] = []

    # Start train mode
    model.train()

    # TODO
    for image, embedding, targets in train_data:
        # Pass data and labels to the correct device
        image = image.to(device)
        embedding = embedding.to(device)
        targets = targets.to(device)

        # Fordward pass
        outputs = model(image, embedding)
        print(outputs.shape, targets.shape)
        loss_value = loss(outputs, targets.squeeze())

        # METRICS
        # Save loss metric
        losses.append(loss_value.item())

        # Save accuracy metric
        accuracy = compute_accuracy(outputs, targets)
        accuracies.append(accuracy.item())

        # Backward pass
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(losses)), float(np.mean(accuracies))


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.

    Return:
        tuple[float, float]: the loss and accuracy in val step
    """

    # TODO

    # TODO
    # Define metric lists
    # Loss
    losses: list[float] = []
    # Accuracy
    accuracies: list[float] = []

    # Start eval mode
    model.eval()

    # TODO
    for image, embedding, targets in val_data:
        
        with torch.no_grad():
            # Pass data and labels to the correct device
            image = image.to(device)
            embedding = embedding.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(image, embedding)
            loss_value = loss(outputs, targets)

        # METRICS
        # Save loss metric
        losses.append(loss_value.item())

        # Save accuracy metric
        accuracy = compute_accuracy(outputs, targets)
        accuracies.append(accuracy.item())

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(losses)), float(np.mean(accuracies))


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        device: device for running operations.

    Returns:
        accuracy of the test data.
    """

    # TODO
    # Define metric lists
    # Accuracies
    accuracies: list[float] = []

    # Start eval mode
    model.eval()

    for image, embedding, targets in test_data:
        
        with torch.no_grad():
            # Pass data and labels to the correct device
            image = image.to(device)
            embedding = embedding.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(image, embedding)

        # METRICS
        # Save accuracy metric
        accuracy = compute_accuracy(outputs, targets)
        accuracies.append(accuracy.item())

    return float(np.mean(accuracies))
