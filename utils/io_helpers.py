import pandas as pd

def carregar_csv(caminho, coluna_texto="description"):
    df = pd.read_csv(caminho)
    df[coluna_texto] = df[coluna_texto].astype(str)
    return df

def salvar_df(df, caminho_saida):
    df.to_csv(caminho_saida, index=False)
    print(f"Arquivo salvo em: {caminho_saida}")

def info_file(file_path):
    df = pd.read_csv(file_path)
    info_df(df)

def info_df(df):
    df['createdAt'] = pd.to_datetime(df['createdAt'], format='mixed')

    menor_data = df['createdAt'].min()
    maior_data = df['createdAt'].max()

    num_linhas = len(df)
    print(f"A base tem {num_linhas} linhas.")

    print(f"A menor data é: {menor_data}")
    print(f"A maior data é: {maior_data}")
