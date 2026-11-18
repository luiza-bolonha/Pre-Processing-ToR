import pandas as pd


def load_csv(path, text_column="description"):
    """Loads a CSV file and ensures the text column is string typed."""
    df = pd.read_csv(path)
    df[text_column] = df[text_column].astype(str)
    return df


def save_df(df, output_path):
    """Saves a pandas DataFrame to CSV format."""
    df.to_csv(output_path, index=False)
    print(f"File saved to: {output_path}")
