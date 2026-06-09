import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─── Configurações ───────────────────────────────────────
DATASET_PATH = "dataset/EuroSAT"
BATCH_SIZE = 32
IMG_SIZE = 64

# ─── Transformações ──────────────────────────────────────
transform_treino = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],
                         std=[0.2038, 0.1367, 0.1148])
])

transform_val_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],
                         std=[0.2038, 0.1367, 0.1148])
])

# ─── Dataset personalizado ───────────────────────────────
class EuroSATDataset(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.transform = transform

        # Mapeamento classe → número
        self.classes = sorted(self.df['ClassName'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_path, row['Filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row['ClassName']]

        if self.transform:
            image = self.transform(image)

        return image, label

# ─── Criar datasets ──────────────────────────────────────
train_dataset = EuroSATDataset(
    csv_path=os.path.join(DATASET_PATH, "train.csv"),
    dataset_path=DATASET_PATH,
    transform=transform_treino
)

val_dataset = EuroSATDataset(
    csv_path=os.path.join(DATASET_PATH, "validation.csv"),
    dataset_path=DATASET_PATH,
    transform=transform_val_test
)

test_dataset = EuroSATDataset(
    csv_path=os.path.join(DATASET_PATH, "test.csv"),
    dataset_path=DATASET_PATH,
    transform=transform_val_test
)

# ─── Criar DataLoaders ───────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ─── Verificação ─────────────────────────────────────────
print(f"Classes: {train_dataset.classes}")
print(f"Treino:     {len(train_dataset)} imagens")
print(f"Validação:  {len(val_dataset)} imagens")
print(f"Teste:      {len(test_dataset)} imagens")

imagens, labels = next(iter(train_loader))
print(f"\nFormato de um batch: {imagens.shape}")
print(f"Labels do batch:     {labels[:8].tolist()}")
print("\n✅ Pré-processamento OK! DataLoaders prontos.")
