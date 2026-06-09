import os
from PIL import Image

dataset_path = "dataset/EuroSAT"
convertidas = 0
erros = 0

for classe in os.listdir(dataset_path):
    pasta = os.path.join(dataset_path, classe)
    if not os.path.isdir(pasta):
        continue

    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".tif"):
            caminho_tif = os.path.join(pasta, arquivo)
            caminho_jpg = caminho_tif.replace(".tif", ".jpg")

            try:
                img = Image.open(caminho_tif).convert("RGB")
                img.save(caminho_jpg, "JPEG", quality=95)
                os.remove(caminho_tif)
                convertidas += 1
            except Exception as e:
                erros += 1

    print(f"Classe {classe} concluida")

print(f"\nConvertidas: {convertidas} imagens")
print(f"Erros: {erros}")
print("Concluido!")