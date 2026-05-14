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
        name: i
        for i, name in enumerate(LABELS)
    }

    labelled = frame.copy()

    labelled["label_id"] = labelled[
        "label"
    ].replace(label_to_index)

    train_df = labelled[
        labelled["split"] == "train"
    ].reset_index(drop=True)

    val_df = labelled[
        labelled["split"] == "val"
    ].reset_index(drop=True)

    test_df = labelled[
        labelled["split"] == "test"
    ].reset_index(drop=True)

    return (
        label_to_index,
        labelled,
        train_df,
        val_df,
        test_df
    )


def image_to_tensor(path: Path) -> torch.Tensor:

    with Image.open(path) as image:

        image = image.convert("RGB")

        image = image.resize((64, 64))

        image_np = np.asarray(
            image,
            dtype=np.float32
        )

    image_np /= 255.0

    tensor = torch.from_numpy(
        image_np
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    return tensor


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

        return self.frame.shape[0]

    def __getitem__(self, index: int):

        row = self.frame.iloc[index]

        path = self.data_root / row[
            "filepath"
        ]

        image_tensor = image_to_tensor(
            path
        )

        label_tensor = torch.tensor(
            int(row["label_id"]),
            dtype=torch.long
        )

        return (
            image_tensor,
            label_tensor
        )


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

    rng = torch.Generator().manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=rng
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

    return (
        batch_images,
        batch_labels
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

    running_loss = 0.0

    running_correct = 0

    total_samples = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        current_batch = labels.size(0)

        running_loss += (
            loss.item() * current_batch
        )

        running_correct += (
            preds == labels
        ).sum().item()

        total_samples += current_batch

    avg_loss = (
        running_loss / total_samples
    )

    avg_acc = (
        running_correct / total_samples
    )

    return avg_loss, avg_acc


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

            batch_size = labels.size(0)

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
            "val_acc": val_acc
        })

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    return (
        history,
        test_loss,
        test_acc
    )