import re
import spacy
import nltk
from spacy.matcher import Matcher
from spacy.util import filter_spans
from utils.io_helpers import carregar_csv

URL_MENCAO_REGEX = re.compile(r"http\S+|www\S+|https\S+|@\w+|#")
EMOJI_SYMBOL_REGEX = re.compile(r"[^\w\s\-áéíóúâêôãõç]", flags=re.UNICODE)

ABREVIACOES = {
    "q": "que", "pq": "porque", "vc": "você", "vcs": "vocês", "tb": "também",
    "blz": "beleza", "hj": "hoje", "amanha": "amanhã", "mt": "muito", "td": "tudo",
    "obg": "obrigado", "vlw": "valeu", "msg": "mensagem", "n": "não", "s": "sim",
    "tbm": "também", "p": "para", "nao": "não"
}

ABREVIACOES_REGEX = re.compile(r'\b(' + '|'.join(map(re.escape, ABREVIACOES.keys())) + r')\b', flags=re.IGNORECASE)

nlp = spacy.load('pt_core_news_lg')

matcher = Matcher(nlp.vocab)
matcher.add("NUM_UNIDADE", [[
    {"LIKE_NUM": True},
    {"LOWER": {"IN": ["mil", "milhão", "milhões", "bilhão", "bilhões"]}, "OP": "?"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"}
]])
matcher.add("SUBSTANTIVOS", [[{"POS": "NOUN"}]])
matcher.add("PROPRIOS", [[{"POS": "PROPN"}]])
matcher.add("ADJETIVOS", [[{"POS": "ADJ"}]])

def substituir_abreviacoes(match):
    palavra = match.group(0).lower()
    return ABREVIACOES.get(palavra, palavra)

def limpar_texto_bruto(texto: str) -> str:
    texto = URL_MENCAO_REGEX.sub(" ", texto)
    texto = ABREVIACOES_REGEX.sub(substituir_abreviacoes, texto)
    texto = EMOJI_SYMBOL_REGEX.sub(" ", texto)
    return texto

def carregar_stopwords_personalizadas():
    with open('stopwords.txt', "r", encoding="utf-8") as f:
        return set(p.strip().lower() for p in f if p.strip())
    
stopwords = set(nltk.corpus.stopwords.words('portuguese'))
#stopwords = carregar_stopwords_personalizadas()

def capitalizar_palavras(texto):
    return ' '.join(p.capitalize() for p in texto.split())

def pre_processar_doc(doc):
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans_filtrados = filter_spans(spans)

    spans_dict = {span.start: span for span in spans_filtrados}
    ent_by_start = {ent.start: ent for ent in doc.ents}

    tokens = []
    i = 0
    while i < len(doc):
        if i in ent_by_start:
            ent = ent_by_start[i]
            tokens.append(capitalizar_palavras(ent.text.lower()))
            i = ent.end
        elif i in spans_dict:
            span = spans_dict[i]
            span_text = span.text.lower()
            if len(span) == 1:
                token = span[0]
                if token.pos_ == "VERB":
                    tokens.append(token.lemma_)
                elif token.is_alpha and token.text.lower() not in stopwords and len(token.text) >= 3:
                    tokens.append(token.text.lower())
            elif len(span_text) >= 3:
                tokens.append(span_text)
            i = span.end
        else:
            i += 1

        tokens_unicos = list(dict.fromkeys(tokens))
    return tokens_unicos

def processar_lote_textos(lista_de_textos, batch_size=1000, n_process=1):
    docs = nlp.pipe([limpar_texto_bruto(txt) for txt in lista_de_textos],
                    batch_size=batch_size,
                    n_process=n_process)
    return [pre_processar_doc(doc) for doc in docs]

def pre_processing_database(file_path, column="description", batch_size=1000, n_process=1):
    df = carregar_csv(file_path,column)
    textos = df[column].tolist()
    resultados = processar_lote_textos(textos, batch_size, n_process)
    return resultados, textos
