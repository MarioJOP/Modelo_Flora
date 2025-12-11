import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def separar_dados(dataset_dir='Tree_Species_Dataset', output_dir='dados_separados', val_ratio=0.2):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Cria as pastas de saída
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val').mkdir(parents=True, exist_ok=True)

    categorias = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f'Encontradas {len(categorias)} categorias.')

    for categoria in categorias:
        imagens = list(categoria.glob('*'))
        treino, val = train_test_split(imagens, test_size=val_ratio, random_state=42)

        # Cria as pastas de treino e validação para cada categoria
        (output_dir / 'train' / categoria.name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'val' / categoria.name).mkdir(parents=True, exist_ok=True)

        # Copia os arquivos
        for img in treino:
            shutil.copy(img, output_dir / 'train' / categoria.name / img.name)
        for img in val:
            shutil.copy(img, output_dir / 'val' / categoria.name / img.name)

        print(f'Categoria {categoria.name}: {len(treino)} treino, {len(val)} validação')

    print(f'\nSeparação concluída. Dados salvos em: {output_dir.resolve()}')

# Executar a função
# separar_dados()
