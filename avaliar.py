import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocessamento import test_loader, test_dataset
from modelos import CNN_Simples, CNN_Profunda

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = test_dataset.classes

# ─── Função de avaliação ─────────────────────────────────
def avaliar_modelo(modelo, caminho_pesos, nome):
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=DEVICE))
    modelo = modelo.to(DEVICE)
    modelo.eval()

    todos_preditos = []
    todos_labels = []

    with torch.no_grad():
        for imagens, labels in test_loader:
            imagens, labels = imagens.to(DEVICE), labels.to(DEVICE)
            saidas = modelo(imagens)
            _, preditos = torch.max(saidas, 1)
            todos_preditos.extend(preditos.cpu().numpy())
            todos_labels.extend(labels.cpu().numpy())

    todos_preditos = np.array(todos_preditos)
    todos_labels = np.array(todos_labels)

    acc = (todos_preditos == todos_labels).mean() * 100
    print(f"\n{'='*60}")
    print(f"Modelo: {nome}")
    print(f"Acurácia no Teste: {acc:.2f}%")
    print(f"{'='*60}")
    print(classification_report(todos_labels, todos_preditos, target_names=CLASSES))

    # ── Matriz de confusão ──
    cm = confusion_matrix(todos_labels, todos_preditos)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Matriz de Confusão — {nome}")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"matriz_{nome}.png")
    plt.show()

    return todos_preditos, todos_labels

# ─── Função para mostrar exemplos ────────────────────────
def mostrar_exemplos(modelo, caminho_pesos, nome):
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=DEVICE))
    modelo = modelo.to(DEVICE)
    modelo.eval()

    acertos, erros = [], []

    with torch.no_grad():
        for imagens, labels in test_loader:
            imagens_gpu = imagens.to(DEVICE)
            saidas = modelo(imagens_gpu)
            _, preditos = torch.max(saidas, 1)
            preditos = preditos.cpu()

            for i in range(len(labels)):
                if preditos[i] == labels[i] and len(acertos) < 6:
                    acertos.append((imagens[i], labels[i].item(), preditos[i].item()))
                if preditos[i] != labels[i] and len(erros) < 6:
                    erros.append((imagens[i], labels[i].item(), preditos[i].item()))
                if len(acertos) >= 6 and len(erros) >= 6:
                    break
            if len(acertos) >= 6 and len(erros) >= 6:
                break

    # ── Plot ──
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    mean = np.array([0.3444, 0.3803, 0.4078])
    std  = np.array([0.2038, 0.1367, 0.1148])

    for i, (img, real, pred) in enumerate(acertos):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"✅ {CLASSES[real]}", fontsize=8)
        axes[0, i].axis("off")

    for i, (img, real, pred) in enumerate(erros):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        axes[1, i].imshow(img_np)
        axes[1, i].set_title(f"❌ Real: {CLASSES[real]}\nPred: {CLASSES[pred]}", fontsize=7)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Acertos", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Erros", fontsize=11, fontweight='bold')

    plt.suptitle(f"Exemplos — {nome}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"exemplos_{nome}.png")
    plt.show()

# ─── Avaliar os dois modelos ─────────────────────────────
pred1, labels1 = avaliar_modelo(CNN_Simples(num_classes=10), "modelo_CNN_Simples.pth", "CNN_Simples")
pred2, labels2 = avaliar_modelo(CNN_Profunda(num_classes=10), "modelo_CNN_Profunda.pth", "CNN_Profunda")

mostrar_exemplos(CNN_Simples(num_classes=10), "modelo_CNN_Simples.pth", "CNN_Simples")
mostrar_exemplos(CNN_Profunda(num_classes=10), "modelo_CNN_Profunda.pth", "CNN_Profunda")