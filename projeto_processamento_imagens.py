"""
Aplicação de filtros de redução de ruído

Autor: [Arian Rios Gutiérrez]
Curso: Processamento de Imagens
Data: 13/09/2025

Este programa aplica três tipos de filtros a uma imagem com ruído:
1. Filtro de média da vizinhança (averaging)
2. Filtro mediano
3. Filtro gaussiano
"""

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

# --- Funções dos Filtros

def adicionar_ruido_gaussiano(imagem, nivel_ruido=0.3):
    ruido = np.random.normal(0, nivel_ruido * 255, imagem.shape)
    imagem_ruidosa = imagem.astype(float) + ruido
    imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255)
    return imagem_ruidosa.astype(np.uint8)

def filtro_media(imagem, tamanho_kernel=5):
    kernel = np.ones((tamanho_kernel, tamanho_kernel)) / (tamanho_kernel ** 2)
    return ndimage.convolve(imagem, kernel, mode='constant').astype(np.uint8)

def filtro_mediano(imagem, tamanho_kernel=5):
    return ndimage.median_filter(imagem, size=tamanho_kernel)

def filtro_gaussiano(imagem, sigma=1.5):
    return ndimage.gaussian_filter(imagem, sigma=sigma).astype(np.uint8)

# --- Função Principal de Execução ---

def executar_projeto():
    print("=" * 50)
    print("PROJETO DE PROCESSAMENTO DE IMAGENS")
    print("=" * 50)

    # 1. Carregar sua imagem original
    try:
        imagem_pil = Image.open("Image-Test.png").convert('L')
        imagem_original = np.array(imagem_pil)
        print(f"✓ Imagem carregada: {imagem_original.shape}")
    except FileNotFoundError:
        print("✗ ERRO: Imagem não encontrada.")
        return

    # 2. Adicionar ruído
    imagem_ruidosa = adicionar_ruido_gaussiano(imagem_original, nivel_ruido=0.2)
    print("✓ Ruído gaussiano adicionado.")

    # 3. Aplicar os filtros
    filtrada_media = filtro_media(imagem_ruidosa)
    filtrada_mediano = filtro_mediano(imagem_ruidosa)
    filtrada_gaussiano = filtro_gaussiano(imagem_ruidosa)
    print("✓ Filtros de média, mediano e gaussiano aplicados.")

    # 4. Visualizar os resultados
    print("✓ Gerando visualização dos resultados...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Dicionário para facilitar a plotagem
    imagens = {
        'Original': imagem_original,
        'Com Ruído': imagem_ruidosa,
        'Filtro Média': filtrada_media,
        'Filtro Mediano': filtrada_mediano,
        'Filtro Gaussiano': filtrada_gaussiano
    }
    
    for i, (titulo, img) in enumerate(imagens.items()):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(titulo)
        axes[i].axis('off') # Ocultar eixos

    plt.tight_layout()
    plt.suptitle("Comparação de Filtros de Redução de Ruído", fontsize=16, y=1.05)
    plt.show()

    print("\n✓ Projeto finalizado com sucesso!")


# --- Ponto de Entrada do Script ---
# Este bloco garante que o código só será executado quando você rodar o arquivo diretamente.
if __name__ == "__main__":
    executar_projeto()
