import os
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "dataset/EuroSAT"
classes = sorted(os.listdir(dataset_path))
classes = [c for c in classes if os.path.isdir(os.path.join(dataset_path, c))]

print("Classes encontradas:")
for classe in classes:
    pasta = os.path.join(dataset_path, classe)
    qtd = len(os.listdir(pasta))
    print(f"  {classe}: {qtd} imagens")

# Mostrar uma imagem de exemplo de cada classe
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, classe in enumerate(classes):
    pasta = os.path.join(dataset_path, classe)
    img_path = os.path.join(pasta, os.listdir(pasta)[0])
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(classe)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("exemplos_dataset.png")
plt.show()
print("Imagem salva como exemplos_dataset.png")