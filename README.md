# 🛰️ Classificador de Imagens de Satélite — EuroSAT

Projeto desenvolvido para a disciplina **Applied Computer Vision (ACV)** da FIAP, com foco na aplicação de **Visão Computacional na Indústria Espacial**. O objetivo é classificar imagens de satélite em 10 categorias de uso do solo utilizando redes neurais convolucionais (CNNs) treinadas do zero com PyTorch.

---

## 👥 Integrantes

| Nome | RM |
|---|---|
| João Pedro de Albuquerque Oliveira | 551579 |
| Pedro Augusto Carneiro Barone Bomfim | 99781 |
| Matheus Augusto Santos Rego | 551466 |

---

## 🎯 Problema

Classificação automática de imagens de satélite em 10 classes de terreno/uso do solo, conectada ao contexto da Indústria Espacial. O modelo é capaz de identificar padrões visuais em imagens capturadas por satélites e categorizá-las com alta precisão.

---

## 📦 Dataset

- **Nome:** EuroSAT
- **Fonte:** [Kaggle — apollo2506/eurosat-dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **Total de imagens:** 27.000
- **Divisão:** 18.900 treino / 5.400 validação / 2.700 teste
- **Tamanho das imagens:** 64x64 pixels (RGB)

### Classes:
| Classe | Imagens |
|---|---|
| AnnualCrop | 3.000 |
| Forest | 3.000 |
| HerbaceousVegetation | 3.000 |
| Highway | 2.500 |
| Industrial | 2.500 |
| Pasture | 2.000 |
| PermanentCrop | 2.500 |
| Residential | 3.000 |
| River | 2.500 |
| SeaLake | 3.000 |

---

## 🧠 Arquiteturas CNN

Foram criadas e treinadas **2 arquiteturas CNN do zero**, sem uso de modelos pré-treinados.

### CNN Simples
- 3 blocos convolucionais (Conv2d + ReLU + MaxPool)
- Camadas densas com Dropout
- **2.193.226 parâmetros**

### CNN Profunda
- 5 blocos convolucionais com BatchNorm e Dropout
- Camadas densas com Dropout mais agressivo
- **4.468.778 parâmetros**

---

## 📊 Resultados

| Modelo | Acurácia no Teste |
|---|---|
| CNN Simples | **94.07%** ✅ |
| CNN Profunda | **92.04%** ✅ |

> Meta mínima exigida: **88%** — ambos os modelos superaram!

A CNN Simples apresentou melhor desempenho, pois o dataset é relativamente simples e uma arquitetura mais leve generalizou melhor.

---

## 🚀 Como executar

### Pré-requisitos

- Python 3.10+
- GPU NVIDIA (recomendado)

### 1. Clone o repositório

```bash
git clone https://github.com/eojotape/projeto-acv.git
cd projeto-acv
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe o dataset

```bash
python -m kaggle datasets download -d apollo2506/eurosat-dataset -p dataset/
```

> Configure sua chave da API do Kaggle antes: [instruções aqui](https://www.kaggle.com/docs/api)

### 4. Execute o pré-processamento

```bash
python preprocessamento.py
```

### 5. Treine os modelos

```bash
python treinar.py
```

### 6. Avalie os modelos

```bash
python avaliar.py
```

### 7. Execute a demonstração

```bash
python demo.py
```

Acesse em: `http://localhost:7860`

---

## 📁 Estrutura do projeto

```
projeto-acv/
├── preprocessamento.py      # Pré-processamento e DataLoaders
├── modelos.py               # Arquiteturas CNN Simples e CNN Profunda
├── treinar.py               # Script de treinamento
├── avaliar.py               # Avaliação e métricas
├── demo.py                  # Interface Gradio
├── requirements.txt         # Dependências do projeto
├── modelo_CNN_Simples.pth   # Pesos do melhor modelo
├── modelo_CNN_Profunda.pth  # Pesos da CNN Profunda
├── graficos_treino.png      # Gráficos de acurácia e loss
├── matriz_CNN_Simples.png   # Matriz de confusão CNN Simples
├── matriz_CNN_Profunda.png  # Matriz de confusão CNN Profunda
└── dataset/
    └── EuroSAT/             # Dataset com 10 classes
```

---

## 🎥 Demonstração

Vídeo de apresentação: *(link do YouTube aqui)*

---

## 🛠️ Tecnologias utilizadas

- Python 3.10+
- PyTorch + TorchVision
- Gradio
- Matplotlib / Seaborn
- Scikit-learn
- Pandas / NumPy / Pillow
