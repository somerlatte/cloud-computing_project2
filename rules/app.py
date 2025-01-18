import pandas as pd
import pickle
import os
from fpgrowth_py import fpgrowth


def process_csv(file_path, chunk_size=200000):
    """
    Processa o arquivo CSV para gerar listas de playlists e um mapeamento entre URIs e nomes das músicas.
    """
    baskets = []
    uri_to_name_map = {}

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk.columns = [col.strip() for col in chunk.columns]

        required_cols = ['pid', 'track_uri', 'track_name']
        if not set(required_cols).issubset(chunk.columns):
            raise KeyError(f"As colunas obrigatórias {required_cols} não foram encontradas no arquivo.")

        uri_to_name_map.update(dict(zip(chunk['track_uri'], chunk['track_name'])))

        playlists = chunk.groupby('pid')['track_uri'].apply(list).tolist()
        baskets.extend(playlists)

    return baskets, uri_to_name_map


def generate_rules(baskets, uri_map, support_threshold=0.05, confidence_threshold=0.5):
    """
    Gera regras de associação a partir das playlists usando o algoritmo FP-Growth.
    """
    frequent_items, raw_rules = fpgrowth(baskets, minSupRatio=support_threshold, minConf=confidence_threshold)

    formatted_rules = [
        {
            'antecedent': [uri_map.get(uri, uri) for uri in rule[0]],
            'consequent': [uri_map.get(uri, uri) for uri in rule[1]],
            'confidence': rule[2]
        }
        for rule in raw_rules
    ]

    return formatted_rules


def save_to_file(data, file_path='/mnt/shared/recommendation_rules.pkl'):
    """
    Salva as regras de associação em um arquivo binário.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_from_file(file_path='/mnt/shared/recommendation_rules.pkl'):
    """
    Carrega regras de associação de um arquivo binário.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def main():
    """
    Função principal para processar dados, gerar regras e salvar os resultados.
    """
    csv_path = os.getenv(
        'CSV_FILENAME',
        'https://raw.githubusercontent.com/somerlatte/cloud-computing_project2/refs/heads/main/data/2023_spotify_ds2.csv'
    )

    print("Iniciando processamento do arquivo CSV...")
    baskets, uri_to_name = process_csv(csv_path)

    print("Gerando regras de associação a partir das playlists...")
    association_rules = generate_rules(baskets, uri_to_name, support_threshold=0.08, confidence_threshold=0.4)

    print("Salvando as regras geradas...")
    save_to_file(association_rules)

    print("Regras geradas e salvas com sucesso!")


if __name__ == '__main__':
    main()
