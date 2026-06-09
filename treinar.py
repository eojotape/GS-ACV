import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocessamento import train_loader, val_loader
from modelos import CNN_Simples, CNN_Profunda

# ─── Configurações ───────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
LR = 0.001

print(f"Usando: {DEVICE}")

# ─── Função de treino ────────────────────────────────────
def treinar_modelo(modelo, nome, epochs=EPOCHS):
    modelo = modelo.to(DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(otimizador, step_size=7, gamma=0.1)

    historico = {"treino_loss": [], "treino_acc": [], "val_loss": [], "val_acc": []}
    melhor_acc = 0.0

    for epoch in range(epochs):
        # ── Treino ──
        modelo.train()
        treino_loss, treino_corretos, treino_total = 0, 0, 0

        for imagens, labels in train_loader:
            imagens, labels = imagens.to(DEVICE), labels.to(DEVICE)
            otimizador.zero_grad()
            saidas = modelo(imagens)
            loss = criterio(saidas, labels)
            loss.backward()
            otimizador.step()

            treino_loss += loss.item()
            _, preditos = torch.max(saidas, 1)
            treino_corretos += (preditos == labels).sum().item()
            treino_total += labels.size(0)

        # ── Validação ──
        modelo.eval()
        val_loss, val_corretos, val_total = 0, 0, 0

        with torch.no_grad():
            for imagens, labels in val_loader:
                imagens, labels = imagens.to(DEVICE), labels.to(DEVICE)
                saidas = modelo(imagens)
                loss = criterio(saidas, labels)

                val_loss += loss.item()
                _, preditos = torch.max(saidas, 1)
                val_corretos += (preditos == labels).sum().item()
                val_total += labels.size(0)

        # ── Métricas ──
        t_loss = treino_loss / len(train_loader)
        t_acc  = treino_corretos / treino_total * 100
        v_loss = val_loss / len(val_loader)
        v_acc  = val_corretos / val_total * 100

        historico["treino_loss"].append(t_loss)
        historico["treino_acc"].append(t_acc)
        historico["val_loss"].append(v_loss)
        historico["val_acc"].append(v_acc)

        print(f"[{nome}] Época {epoch+1:02d}/{epochs} | "
              f"Treino Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")

        # ── Salvar melhor modelo ──
        if v_acc > melhor_acc:
            melhor_acc = v_acc
            torch.save(modelo.state_dict(), f"modelo_{nome}.pth")
            print(f"  ✅ Melhor modelo salvo! Val Acc: {v_acc:.2f}%")

        scheduler.step()

    print(f"\n🏆 Melhor acurácia de validação [{nome}]: {melhor_acc:.2f}%\n")
    return historico

# ─── Treinar os dois modelos ─────────────────────────────
print("=" * 60)
print("Treinando CNN Simples...")
print("=" * 60)
hist1 = treinar_modelo(CNN_Simples(num_classes=10), "CNN_Simples")

print("=" * 60)
print("Treinando CNN Profunda...")
print("=" * 60)
hist2 = treinar_modelo(CNN_Profunda(num_classes=10), "CNN_Profunda")

# ─── Gráficos ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, EPOCHS + 1)

axes[0, 0].plot(epochs_range, hist1["treino_acc"], label="Treino")
axes[0, 0].plot(epochs_range, hist1["val_acc"], label="Validação")
axes[0, 0].set_title("CNN Simples — Acurácia")
axes[0, 0].set_xlabel("Época")
axes[0, 0].set_ylabel("Acurácia (%)")
axes[0, 0].legend()

axes[0, 1].plot(epochs_range, hist1["treino_loss"], label="Treino")
axes[0, 1].plot(epochs_range, hist1["val_loss"], label="Validação")
axes[0, 1].set_title("CNN Simples — Loss")
axes[0, 1].set_xlabel("Época")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()

axes[1, 0].plot(epochs_range, hist2["treino_acc"], label="Treino")
axes[1, 0].plot(epochs_range, hist2["val_acc"], label="Validação")
axes[1, 0].set_title("CNN Profunda — Acurácia")
axes[1, 0].set_xlabel("Época")
axes[1, 0].set_ylabel("Acurácia (%)")
axes[1, 0].legend()

axes[1, 1].plot(epochs_range, hist2["treino_loss"], label="Treino")
axes[1, 1].plot(epochs_range, hist2["val_loss"], label="Validação")
axes[1, 1].set_title("CNN Profunda — Loss")
axes[1, 1].set_xlabel("Época")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("graficos_treino.png")
plt.show()
print("✅ Gráficos salvos em graficos_treino.png")