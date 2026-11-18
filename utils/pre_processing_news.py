"""Preprocessing for journalistic news articles (Portuguese)."""

import csv

import nltk
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load("pt_core_news_lg")

try:
    nltk.corpus.stopwords.words("portuguese")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass
stopwords = set(nltk.corpus.stopwords.words("portuguese"))

# Matcher for quantity expressions such as "20 mil pessoas", "vinte pessoas", "primeiro dia"
matcher = Matcher(nlp.vocab)
cardinal_noun_pattern = [
    {"LIKE_NUM": True},
    {"LOWER": {"IN": ["mil", "milhão", "milhões", "bilhão", "bilhões"]}, "OP": "?"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"},
]
spelled_number_pattern = [
    {"POS": "NUM"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"},
]
ordinal_pattern = [
    {"TAG": "ORD"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"},
]
matcher.add("NUM_UNIT", [cardinal_noun_pattern, spelled_number_pattern, ordinal_pattern])


def _is_stopword(text):
    return text.lower() in stopwords


def preprocess_text(text):
    """
    Tokenizes a text keeping multi-word units as single tokens:
    quantity expressions (matcher) and named entities. Other tokens are kept
    if alphabetic and not stopwords. Returns unique tokens in order.
    """
    doc = nlp(text)

    # Index spans by their start token, so each token is consumed only once
    quantity_spans = {span.start: span for span in filter_spans([doc[s:e] for _, s, e in matcher(doc)])}
    entities = {ent.start: ent for ent in doc.ents}

    tokens = []
    i = 0
    while i < len(doc):
        span = quantity_spans.get(i) or entities.get(i)
        if span is not None:
            if not _is_stopword(span.text):
                tokens.append(span.text)
            i = span.end
        else:
            token = doc[i]
            if token.is_alpha and not _is_stopword(token.text):
                tokens.append(token.text)
            i += 1

    return list(dict.fromkeys(tokens))


def preprocess_text_simple(text):
    """Single-word tokenization: alphabetic, non-stopword tokens (unique, in order)."""
    tokens = [t.text for t in nlp(text) if t.is_alpha and not _is_stopword(t.text)]
    return list(dict.fromkeys(tokens))


def pre_processing_database(file_path, split_paragraphs, column="Texto", simple=False):
    """
    Reads the news CSV and tokenizes the given column.
    Returns (transactions, original_texts). With split_paragraphs, each
    paragraph (line) becomes its own transaction.
    """
    blocks = []
    news = []

    with open(file_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row[column]
            news.append(text)
            if split_paragraphs:
                blocks.extend(p.strip() for p in text.split("\n") if p.strip())
            else:
                blocks.append(text)

    tokenize = preprocess_text_simple if simple else preprocess_text
    return [tokenize(block) for block in blocks], news
