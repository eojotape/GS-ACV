import torch
import torch.nn as nn

# ─── CNN 1 — Simples ─────────────────────────────────────
class CNN_Simples(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Simples, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 → 32x32

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 → 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── CNN 2 — Profunda ────────────────────────────────────
class CNN_Profunda(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Profunda, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 → 32x32
            nn.Dropout(0.2),

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16
            nn.Dropout(0.2),

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 → 8x8
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── Verificação ─────────────────────────────────────────
if __name__ == "__main__":
    modelo1 = CNN_Simples(num_classes=10)
    modelo2 = CNN_Profunda(num_classes=10)

    x = torch.randn(1, 3, 64, 64)

    out1 = modelo1(x)
    out2 = modelo2(x)

    total1 = sum(p.numel() for p in modelo1.parameters())
    total2 = sum(p.numel() for p in modelo2.parameters())

    print("─── CNN Simples ───────────────────────")
    print(f"Saída: {out1.shape}")
    print(f"Parâmetros: {total1:,}")

    print("\n─── CNN Profunda ──────────────────────")
    print(f"Saída: {out2.shape}")
    print(f"Parâmetros: {total2:,}")

    print("\n✅ Modelos criados com sucesso!")