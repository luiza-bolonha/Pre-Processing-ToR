"""Dataset utilities: summary info, date filtering and near-duplicate removal."""

import pandas as pd
from tqdm import tqdm

DATE_COLUMN = "createdAt"

_nlp = None


def _get_nlp():
    """Loads the Portuguese SpaCy model on demand (large, falling back to small)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("pt_core_news_lg")
        except Exception:
            _nlp = spacy.load("pt_core_news_sm")
    return _nlp


def info_file(file_path):
    """Prints the row count and date range of a CSV file."""
    info_df(pd.read_csv(file_path))


def info_df(df):
    """Prints the row count and date range of a DataFrame."""
    dates = pd.to_datetime(df[DATE_COLUMN], format="mixed")
    print(f"The dataset has {len(df)} rows.")
    print(f"Earliest date: {dates.min()}")
    print(f"Latest date: {dates.max()}")


def cut_database_by_date(original_file, start_date, end_date):
    """Returns the rows with start_date <= createdAt < end_date, sorted by date."""
    df = pd.read_csv(original_file)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="mixed")
    df = df.sort_values(by=DATE_COLUMN)
    in_range = (df[DATE_COLUMN] >= pd.Timestamp(start_date)) & (df[DATE_COLUMN] < pd.Timestamp(end_date))
    return df[in_range]


def remove_similar_texts(file, column="description", threshold=0.92):
    """
    Removes near-duplicate texts using SpaCy vector similarity:
    a text is dropped if it is >= `threshold` similar to one already kept.
    """
    df = pd.read_csv(file)
    print(f"Original file size: {len(df)}")

    nlp = _get_nlp()
    docs = list(nlp.pipe(df[column].astype(str).tolist(), batch_size=1000))

    kept = []
    for i, doc in enumerate(tqdm(docs, desc="Checking similarity")):
        if all(doc.similarity(docs[j]) < threshold for j in kept):
            kept.append(i)

    df_filtered = df.iloc[kept].reset_index(drop=True)
    print(f"Rows removed: {len(df) - len(df_filtered)}")
    print(f"Processed file size: {len(df_filtered)}")
    return df_filtered
