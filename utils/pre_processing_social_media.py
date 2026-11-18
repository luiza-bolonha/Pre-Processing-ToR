"""Preprocessing for social media posts / tweets (Portuguese)."""

import re

import nltk
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

from utils.io_helpers import load_csv

URL_MENTION_REGEX = re.compile(r"http\S+|www\S+|https\S+|@\w+|#")
EMOJI_SYMBOL_REGEX = re.compile(r"[^\w\s\-áéíóúâêôãõç]", flags=re.UNICODE)

# Common Portuguese internet abbreviations -> full word
ABBREVIATIONS = {
    "q": "que", "pq": "porque", "vc": "você", "vcs": "vocês", "tb": "também",
    "blz": "beleza", "hj": "hoje", "amanha": "amanhã", "mt": "muito", "td": "tudo",
    "obg": "obrigado", "vlw": "valeu", "msg": "mensagem", "n": "não", "s": "sim",
    "tbm": "também", "p": "para", "nao": "não",
}
ABBREVIATIONS_REGEX = re.compile(
    r"\b(" + "|".join(map(re.escape, ABBREVIATIONS)) + r")\b", flags=re.IGNORECASE
)

MIN_TOKEN_LENGTH = 3

nlp = spacy.load("pt_core_news_lg")

# Full matcher: quantity expressions + nouns, proper nouns and adjectives
matcher = Matcher(nlp.vocab)
matcher.add("NUM_UNIT", [[
    {"LIKE_NUM": True},
    {"LOWER": {"IN": ["mil", "milhão", "milhões", "bilhão", "bilhões"]}, "OP": "?"},
    {"LOWER": "de", "OP": "?"},
    {"POS": "NOUN"},
]])
matcher.add("NOUNS", [[{"POS": "NOUN"}]])
matcher.add("PROPER_NOUNS", [[{"POS": "PROPN"}]])
matcher.add("ADJECTIVES", [[{"POS": "ADJ"}]])

# Simple matcher: nouns, proper nouns and adjectives only
simple_matcher = Matcher(nlp.vocab)
simple_matcher.add("NOUNS", [[{"POS": "NOUN"}]])
simple_matcher.add("PROPER_NOUNS", [[{"POS": "PROPN"}]])
simple_matcher.add("ADJECTIVES", [[{"POS": "ADJ"}]])

stopwords = set(nltk.corpus.stopwords.words("portuguese"))


def load_custom_stopwords(path="stopwords.txt"):
    """Loads one stopword per line from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def _expand_abbreviation(match):
    word = match.group(0).lower()
    return ABBREVIATIONS.get(word, word)


def clean_raw_text(text: str) -> str:
    """Removes URLs, mentions, '#', emojis/symbols and expands abbreviations."""
    text = URL_MENTION_REGEX.sub(" ", text)
    text = ABBREVIATIONS_REGEX.sub(_expand_abbreviation, text)
    return EMOJI_SYMBOL_REGEX.sub(" ", text)


def capitalize_words(text):
    return " ".join(word.capitalize() for word in text.split())


def _matched_spans(doc, span_matcher):
    """Returns non-overlapping matcher spans indexed by start token."""
    spans = filter_spans([doc[start:end] for _, start, end in span_matcher(doc)])
    return {span.start: span for span in spans}


def _single_token_value(token, lemmatize_verbs):
    """Value for a one-token span: verb lemma, or the lowercase word if it is long and not a stopword."""
    if lemmatize_verbs and token.pos_ == "VERB":
        return token.lemma_
    text = token.text.lower()
    if token.is_alpha and text not in stopwords and len(text) >= MIN_TOKEN_LENGTH:
        return text
    return None


def _span_tokens(doc, span_matcher, use_entities, lemmatize_verbs):
    """
    Collects tokens from named entities (capitalized) and matcher spans.
    Tokens outside entities/spans are discarded. Returns unique tokens in order.
    """
    spans = _matched_spans(doc, span_matcher)
    entities = {ent.start: ent for ent in doc.ents} if use_entities else {}

    tokens = []
    i = 0
    while i < len(doc):
        if i in entities:
            ent = entities[i]
            tokens.append(capitalize_words(ent.text.lower()))
            i = ent.end
        elif i in spans:
            span = spans[i]
            if len(span) == 1:
                value = _single_token_value(span[0], lemmatize_verbs)
            else:
                value = span.text.lower() if len(span.text) >= MIN_TOKEN_LENGTH else None
            if value:
                tokens.append(value)
            i = span.end
        else:
            i += 1

    return list(dict.fromkeys(tokens))


def preprocess_doc(doc):
    """Full tokenization: entities, quantity expressions, nouns, proper nouns and adjectives."""
    return _span_tokens(doc, matcher, use_entities=True, lemmatize_verbs=True)


def preprocess_doc_simple(doc):
    """Simple tokenization: nouns, proper nouns and adjectives only."""
    return _span_tokens(doc, simple_matcher, use_entities=False, lemmatize_verbs=False)


def process_texts(texts, batch_size=1000, n_process=1, simple=False):
    """Cleans and tokenizes a list of texts with SpaCy in batches."""
    docs = nlp.pipe((clean_raw_text(t) for t in texts), batch_size=batch_size, n_process=n_process)
    preprocess = preprocess_doc_simple if simple else preprocess_doc
    return [preprocess(doc) for doc in docs]


def pre_processing_database(file_path, column="description", batch_size=1000, n_process=1, simple=False):
    """Reads the CSV and returns (transactions, original_texts)."""
    texts = load_csv(file_path, column)[column].tolist()
    return process_texts(texts, batch_size, n_process, simple), texts
