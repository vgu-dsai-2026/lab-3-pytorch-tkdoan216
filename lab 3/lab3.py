import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("data")
BATCH_SIZE = 32
SEED = 42
EPOCHS = 5
LABELS = ["cat", "dog"]

df = pd.DataFrame({
    "filepath": [
        "cat1.jpg",
        "dog1.jpg",
        "cat2.jpg",
        "dog2.jpg"
    ],
    "label": [
        "cat",
        "dog",
        "cat",
        "dog"
    ]
})

label_to_index = {
    "cat": 0,
    "dog": 1
}

df["label_id"] = df["label"].map(label_to_index)

df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train_df = df[:train_size]

val_df = df[
    train_size:train_size + val_size
]

test_df = df[
    train_size + val_size:
]

def image_to_tensor(path: Path) -> torch.Tensor:

    image = Image.open(path).convert("RGB")

    image = image.resize((64, 64))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    image_tensor = image_tensor.permute(2, 0, 1)

    return image_tensor

class CatsDogsDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path
    ):

        self.frame = frame.reset_index(drop=True)

        self.data_root = data_root

    def __len__(self):

        return len(self.frame)

    def __getitem__(self, index):

        row = self.frame.iloc[index]

        image_path = self.data_root / row["filepath"]

        image_tensor = image_to_tensor(image_path)

        label_tensor = torch.tensor(
            row["label_id"],
            dtype=torch.long
        )

        return image_tensor, label_tensor

train_loader_generator = torch.Generator().manual_seed(SEED)

def build_dataloaders(
    train_df,
    val_df,
    test_df,
    data_root,
    batch_size=32,
    seed=SEED,
    dataset_cls=CatsDogsDataset,
):

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_loader_generator
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

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = build_dataloaders(
    train_df,
    val_df,
    test_df,
    DATA_ROOT,
    batch_size=BATCH_SIZE,
)

def inspect_first_batch(loader):

    if loader is None:
        raise ValueError(
            "Complete Question 3 before inspecting a batch."
        )

    batch_images, batch_labels = next(iter(loader))

    print(batch_images.shape)
    print(batch_images.dtype)

    print(batch_labels.shape)
    print(batch_labels.dtype)

    assert batch_images.ndim == 4
    assert batch_images.shape[1] == 3
    assert batch_labels.dtype == torch.long

    return batch_images, batch_labels

batch_images, batch_labels = inspect_first_batch(
    train_loader
)

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

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.classifier(x)

        return x

model = CatsDogsSimpleCNN()

example_logits = model(batch_images[:4])

print(example_logits.shape)

assert example_logits.shape == (4, 2)

def setup_training(
    model,
    device=None,
    learning_rate=1e-3,
):

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

device, model, criterion, optimizer = setup_training(
    model
)

print(device)

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

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

        total_loss += loss.item() * batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    average_loss = total_loss / total_examples

    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy

def evaluate(
    model,
    loader,
    criterion,
    device,
):

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

            total_loss += loss.item() * batch_size

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples

    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy

val_loss, val_acc = evaluate(
    model,
    val_loader,
    criterion,
    device
)

print(
    f"Validation: loss={val_loss:.4f}, acc={val_acc:.3f}"
)

def run_training_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
    plot=True,
):

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

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    if plot:

        train_losses = [
            h["train_loss"]
            for h in history
        ]

        val_losses = [
            h["val_loss"]
            for h in history
        ]

        plt.plot(
            train_losses,
            label="Train Loss"
        )

        plt.plot(
            val_losses,
            label="Validation Loss"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title(
            "Training vs Validation Loss"
        )

        plt.legend()

        plt.show()

    return history, test_loss, test_acc

history, test_loss, test_acc = run_training_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=EPOCHS,
    plot=True,
)

print(history)