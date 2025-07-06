import pandas as pd

def carregar_csv(caminho, coluna_texto="description"):
    df = pd.read_csv(caminho)
    df[coluna_texto] = df[coluna_texto].astype(str)
    return df

def salvar_df(df, caminho_saida):
    df.to_csv(caminho_saida, index=False)
    print(f"Arquivo salvo em: {caminho_saida}")