# 🌿 Modelo_Flora — Classificação de Espécies Arbóreas com Deep Learning

Este repositório implementa um pipeline completo de **classificação de espécies de árvores** utilizando **Transfer Learning com PyTorch**.  
O projeto faz parte do sistema **ItapajéBio**, voltado para o reconhecimento automatizado da flora local do município de Itapajé (UFC Campus Jardins de Anita).

---

## 🧩 Estrutura do Projeto

```
Modelo_Flora/
│
├── dados/
│   ├── Tree_Species_Dataset/        # Dataset original com pastas por classe
│   └── dados_separados/             # Dataset dividido em treino e validação
│       ├── train/
│       └── val/
│
├── utils/
│   ├── data.py                      # Criação dos DataLoaders e transformações
│   ├── model.py                     # Construção e salvamento do modelo (VGG16, ResNet50, MobileNetV2)
│   ├── train_and_valid.py           # Rotina de treino e validação
│   ├── plot_history.py              # Gráficos de acurácia e perda por época
│   ├── separar_dados.py             # Script para dividir dataset em treino/validação
│   └── show_images.py               # Exibição de imagens transformadas
│
├── main.py                          # Script principal: orquestra treino e avaliação
├── requirements.txt / pyproject.toml # Dependências do projeto
└── uv.lock                         
```

---

## ⚙️ Requisitos

O projeto foi desenvolvido para um ambiente de **Deep Learning com GPU** seguindo as recomendações técnicas da disciplina **TL1 - Deep Learning**.

### Software
- Python 3.12  
- PyTorch e Torchvision  
- Matplotlib  
- Scikit-learn  
- uv (gerenciador de dependências)

### Hardware
- CPU: 8 núcleos (Intel i7/Ryzen 7 ou superior)  
- RAM: 16 GB ou mais  
- GPU: NVIDIA com suporte a CUDA (ex. RTX 3060 8GB)  
- SO recomendado: Ubuntu 20.04+

---

## 📦 Instalação

Clone o repositório e crie o ambiente virtual com o [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MarioJOP/Modelo_Flora.git
cd Modelo_Flora
uv sync
```

Ou, com `pip` tradicional:

```bash
git clone https://github.com/MarioJOP/Modelo_Flora.git
cd Modelo_Flora
pip install -r requirements.txt
```

---

## 🌱 Preparação dos Dados

1. **Coloque as imagens originais** no diretório `dados/Tree_Species_Dataset/`,  
   com cada espécie em uma subpasta (ex.: `bamboo/`, `mango/`, `banyan/` etc).

2. **Divida automaticamente** em treino e validação executando:

```bash
python -m utils.separar_dados
```

Isso criará `dados/dados_separados/train` e `dados/dados_separados/val`, mantendo a proporção 80/20.

---

## 🧠 Treinamento do Modelo

O script principal `main.py` realiza:
- carregamento dos dados (`create_data_loaders`);
- construção do modelo base (`build_model`);
- treinamento e validação (`train_model`);
- visualização dos gráficos (`plot_history`).

Para treinar:

```bash
python main.py
```

Durante a execução, serão exibidas:
- métricas por época (loss e accuracy);
- gráficos de aprendizado por otimizador e taxa de aprendizado.

Por padrão, o modelo base é **ResNet50**, mas pode ser alterado para `"vgg16"` ou `"mobilenetv2"` no arquivo `main.py`:
```python
base_name = "resnet50"
```

---

## 🌼 Visualização de Imagens

Para inspecionar as transformações aplicadas (rotação, cor, etc.), use:

```python
from utils.show_images import show_random_images
show_random_images(train_loader, classes)
```

---

## 🧪 Arquitetura do Modelo

O projeto aplica **Transfer Learning** com modelos pré-treinados no **ImageNet**, adaptando a última camada para as classes de árvores.  
Transformações incluem:
- `RandomResizedCrop` (zoom);
- `RandomRotation(20°)`;
- `RandomHorizontalFlip`;
- `ColorJitter` (iluminação e saturação);
- `GaussianBlur` opcional.

Essas augmentations simulam variações naturais de campo, como iluminação, vento e ângulos de captura.

---

## 📊 Resultados e Monitoramento

O histórico de treino/validação é salvo em memória e pode ser visualizado com `plot_history.py`, gerando gráficos de **acurácia e perda** para análise do desempenho.

Exemplo de saída:
```
Epoch 10/50 | Train Loss: 0.3251 Acc: 0.9024 | Val Loss: 0.2876 Acc: 0.9147
```

---

## 💾 Salvamento do Modelo

Durante ou após o treinamento, os pesos podem ser salvos com:
```python
from utils.model import save_model
save_model(model, optimizer, epoch=50, val_acc=0.91, prefix="resnet50")
```

Isso criará um arquivo `.pth` em `models/` contendo:
- parâmetros do modelo,
- estado do otimizador,
- acurácia final de validação.

---

## 🌍 Contexto Acadêmico

Este projeto integra a frente de **classificação da flora arbórea** do sistema **ItapajéBio**, descrito no documento *Deep Learning – TL1 (Turma CD VI)*.  
O objetivo é apoiar o mapeamento e identificação da biodiversidade regional por meio de inteligência artificial e aprendizado profundo, alinhado às iniciativas de extensão e pesquisa da **Universidade Federal do Ceará – Campus Itapajé**.

---

## 👩‍💻 Autoria

**Turma CD VI - 2025**  
UFC – Campus Itapajé  
Projeto ItapajéBio (Classificação de Flora e Sons de Pássaros)
