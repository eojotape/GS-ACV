import torch
import gradio as gr
import numpy as np
from PIL import Image
import tifffile
from torchvision import transforms
from modelos import CNN_Simples

# ─── Configurações ───────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# ─── Carregar modelo ─────────────────────────────────────
modelo = CNN_Simples(num_classes=10)
modelo.load_state_dict(torch.load("modelo_CNN_Simples.pth", map_location=DEVICE))
modelo = modelo.to(DEVICE)
modelo.eval()

# ─── Transformação ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],
                         std=[0.2038, 0.1367, 0.1148])
])

# ─── Função de predição ──────────────────────────────────
def classificar(caminho_imagem):
    if caminho_imagem is None:
        return {}

    try:
        # Tenta abrir normalmente (jpg, png, etc)
        img = Image.open(caminho_imagem).convert("RGB")
    except Exception:
        # Se falhar, tenta como TIFF multiband
        dados = tifffile.imread(caminho_imagem)
        # Pega apenas os 3 primeiros canais (RGB)
        if dados.ndim == 3 and dados.shape[2] >= 3:
            dados = dados[:, :, :3]
        # Normaliza para 0-255
        dados = dados.astype(np.float32)
        dados = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8) * 255
        img = Image.fromarray(dados.astype(np.uint8)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        saida = modelo(img_tensor)
        probs = torch.softmax(saida, dim=1)[0].cpu().numpy()

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# ─── Interface Gradio ────────────────────────────────────
demo = gr.Interface(
    fn=classificar,
    inputs=gr.Image(label="Envie uma imagem de satelite", type="filepath"),
    outputs=gr.Label(num_top_classes=5, label="Classificacao"),
    title="Classificador de Imagens de Satelite - EuroSAT",
    description="Envie uma imagem de satelite e o modelo ira classificar o tipo de terreno. Modelo: CNN Simples (Acuracia: 94.07%)",
)

demo.launch()