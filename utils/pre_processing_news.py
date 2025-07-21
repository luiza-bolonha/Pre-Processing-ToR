import spacy
import csv
import nltk
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load('pt_core_news_lg')

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))
#stopwords = carregar_stopwords_personalizadas()

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

def pre_processar_txt(texto):
    doc = nlp(texto)

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


def pre_processar_txt_simples(texto):
    doc = nlp(texto)

    tokens = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in stopwords:
            tokens.append(token.text)

    # Remove duplicados preservando a ordem
    tokens_unicos = list(dict.fromkeys(tokens))
    return tokens_unicos

def pre_processing_database(file_path, separar_paragrafos, column="Texto", simples=False):
    valores_coluna = []
    news = []

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            texto = line[column]
            news.append(texto)
            if(separar_paragrafos):
                paragrafos = texto.split('\n')  
                paragrafos = [p.strip() for p in paragrafos if p.strip()]
                valores_coluna.extend(paragrafos)             
            else: 
                valores_coluna.append(texto)
        
    valores = []
    for texto in valores_coluna:
        if simples:
            result = pre_processar_txt_simples(texto)
        else:
            result = pre_processar_txt(texto)    
        valores.append(result)

    return valores, news