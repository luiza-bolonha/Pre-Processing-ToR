import spacy
from tqdm import tqdm
import pandas as pd

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

def cut_database_by_date(original_file,start_date,end_date):
    df = pd.read_csv(original_file)

    df['createdAt'] = pd.to_datetime(df['createdAt'], format='mixed')
    df_ordenado = df.sort_values(by='createdAt')

    data_inicio = pd.Timestamp(start_date)
    data_fim = pd.Timestamp(end_date)
    df_new_database = df_ordenado[(df_ordenado['createdAt'] >= data_inicio) & (df_ordenado['createdAt'] < data_fim)]

    return df_new_database


nlp = spacy.load('pt_core_news_lg')

def remover_textos_semelhantes_spacy(file, column="description", threshold=0.92):
    df = pd.read_csv(file)

    print(f"Tamanho arquivo original: {len(df)}")

    textos = df[column].astype(str).tolist()
    docs = list(nlp.pipe(textos, batch_size=1000))

    indices_para_manter = []
    indices_vistos = []
    
    for i, doc_i in enumerate(tqdm(docs, desc="Verificando similaridade")):
        duplicado = False
        for j in indices_vistos:
            if doc_i.similarity(docs[j]) >= threshold:
                duplicado = True
                break
        if not duplicado:
            indices_para_manter.append(i)
            indices_vistos.append(i)
    
    total_removidos = len(df) - len(df_filtrado)
    print(f"Índices removidos: {total_removidos}")
    print(f"Tamanho arquivo processado: {len(df_filtrado)}")
        
    df_filtrado = df.iloc[indices_para_manter].reset_index(drop=True)
    return df_filtrado
