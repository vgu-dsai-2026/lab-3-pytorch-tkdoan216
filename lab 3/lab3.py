import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


def build_label_mapping(
    frame: pd.DataFrame
) -> tuple[
    dict[str, int],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:

    label_to_index = {
        label: idx
        for idx, label in enumerate(LABELS)
    }

    labelled = frame.copy()

    labelled["label_id"] = labelled[
        "label"
    ].map(label_to_index)

    labelled = labelled.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    train_size = int(len(labelled) * 0.7)

    val_size = int(len(labelled) * 0.15)

    train_df = labelled.iloc[:train_size]

    val_df = labelled.iloc[
        train_size:train_size + val_size
    ]

    test_df = labelled.iloc[
        train_size + val_size:
    ]

    return (
        label_to_index,
        labelled,
        train_df,
        val_df,
        test_df
    )


def image_to_tensor(path: Path) -> torch.Tensor:

    image = Image.open(path).convert("RGB")

    image = image.resize((64, 64))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_tensor = torch.from_numpy(
        image_array
    )

    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )

    return image_tensor


class CatsDogsDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path
    ):

        self.frame = frame.reset_index(
            drop=True
        )

        self.data_root = data_root

    def __len__(self) -> int:

        return len(self.frame)

    def __getitem__(self, index: int):

        row = self.frame.iloc[index]

        image_path = (
            self.data_root /
            row["filepath"]
        )

        image_tensor = image_to_tensor(
            image_path
        )

        label_tensor = torch.tensor(
            row["label_id"],
            dtype=torch.long
        )

        return image_tensor, label_tensor


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_root: Path,
    batch_size: int = 32,
    seed: int = SEED,
    dataset_cls: type[Dataset] = CatsDogsDataset,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader
]:

    train_dataset = dataset_cls(
        train_df,
        data_root
    )

    val_dataset = dataset_cls(
        val_df,
        data_root
    )

    test_dataset = dataset_cls(
        test_df,
        data_root
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


def inspect_first_batch(
    loader: DataLoader
) -> tuple[
    torch.Tensor,
    torch.Tensor
]:

    if loader is None:
        raise ValueError(
            "Complete Question 3 before inspecting a batch."
        )

    batch_images, batch_labels = next(
        iter(loader)
    )

    print(
        "Image batch:",
        batch_images.shape,
        batch_images.dtype
    )

    print(
        "Label batch:",
        batch_labels.shape,
        batch_labels.dtype
    )

    assert batch_images.ndim == 4

    assert batch_images.shape[1] == 3

    assert batch_labels.dtype == torch.long

    return batch_images, batch_labels


class CatsDogsSimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 16 * 16,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                2
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.classifier(x)

        return x


def setup_training(
    model: nn.Module,
    device: torch.device | None = None,
    learning_rate: float = 1e-3,
) -> tuple[
    torch.device,
    nn.Module,
    nn.Module,
    torch.optim.Optimizer
]:

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    return (
        device,
        model,
        criterion,
        optimizer
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_examples = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    average_loss = (
        total_loss / total_examples
    )

    average_accuracy = (
        total_correct / total_examples
    )

    return (
        average_loss,
        average_accuracy
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_examples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = (
        total_loss / total_examples
    )

    average_accuracy = (
        total_correct / total_examples
    )

    return (
        average_loss,
        average_accuracy
    )


def run_training_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    plot: bool = True,
) -> tuple[
    list[dict[str, float]],
    float,
    float,
    float | None
]:

    history = []

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    baseline_acc = None

    return (
        history,
        test_loss,
        test_acc,
        baseline_acc
    )