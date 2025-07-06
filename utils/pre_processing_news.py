import re
import spacy
import nltk
from spacy.matcher import Matcher
from spacy.util import filter_spans
from utils.io_helpers import carregar_csv
from utils.pre_processing_social_media import carregar_stopwords_personalizadas


nlp = spacy.load('pt_core_news_lg')

#nltk.download('stopwords')
#stopwords = set(nltk.corpus.stopwords.words('portuguese'))
stopwords = carregar_stopwords_personalizadas()

# Matcher para padrões como "20 mil pessoas" e datas
matcher = Matcher(nlp.vocab)
padrao_cardinal_substantivo = [
    {"LIKE_NUM": True},
    {"LOWER": {"IN": ["mil", "milhão", "milhões", "bilhão", "bilhões"]}, "OP": "?"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"}
]
padrao_num_extenso = [
    {"POS": "NUM"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"}
]
padrao_ordinal = [
    {"TAG": "ORD"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"}
]
matcher.add("NUM_UNIDADE", [padrao_cardinal_substantivo, padrao_num_extenso, padrao_ordinal])

def pre_processar_doc(doc):
    
    # Identifica spans compostos (ex: "vinte mil pessoas")
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans_filtrados = filter_spans(spans)

    # Mapeia início do span -> span para evitar tokenização duplicada
    spans_dict = {span.start: span for span in spans_filtrados}
    ent_by_start = {ent.start: ent for ent in doc.ents}

    tokens = []
    i = 0
    while i < len(doc):
        if i in spans_dict:
            span = spans_dict[i]
            span_text = span.text
            if span_text.lower() not in stopwords:
                tokens.append(span_text)
            i = span.end
        elif i in ent_by_start:
            ent = ent_by_start[i]
            ent_text = ent.text
            if ent_text.lower() not in stopwords:
                tokens.append(ent_text)
            i = ent.end
        else:
            token = doc[i]
            if token.is_alpha and token.text.lower() not in stopwords:
                tokens.append(token.text)
            i += 1

        tokens_unicos = list(dict.fromkeys(tokens))
    return tokens_unicos

def processar_lote_textos(lista_de_textos, batch_size=1000, n_process=1):
    docs = nlp.pipe(lista_de_textos,
                    batch_size=batch_size,
                    n_process=n_process)
    return [pre_processar_doc(doc) for doc in docs]

def pre_processing_database(file_path, column="Texto", batch_size=1000, n_process=1):
    df = carregar_csv(file_path,column)
    textos = df[column].tolist()
    resultados = processar_lote_textos(textos, batch_size, n_process)
    return resultados, textos

def concordance(list_values, termo, largura=40, case_sensitive=False, show_result=True):
    texto = " ".join(list_values).lower()
    
    if not case_sensitive:
        texto_proc = texto.lower()
        termo_proc = termo.lower()
    else:
        texto_proc = texto
        termo_proc = termo

    ocorrencias = [m.start() for m in re.finditer(re.escape(termo_proc), texto_proc)]
    resultados = []

    for i in ocorrencias:
        inicio = max(0, i - largura)
        fim = min(len(texto), i + len(termo) + largura)

        esquerda = texto[inicio:i].replace('\n', ' ')
        centro = texto[i:i+len(termo)]
        direita = texto[i+len(termo):fim].replace('\n', ' ')

        # Calcula quantos espaços são necessários para alinhar o termo na coluna `largura`
        padding = largura - len(esquerda)
        padding = max(0, padding)

        linha = f"{' ' * padding}{esquerda}{centro}{direita}"
        resultados.append(linha)

    print(f"Número de ocorrências: {len(resultados)}\n")
    for r in resultados:
        print(r)