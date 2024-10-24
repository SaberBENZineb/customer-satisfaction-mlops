import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv("./data/olist_customers_dataset.csv")
            if df.empty:
                raise ValueError("The ingested DataFrame is empty.")
            return df
        except FileNotFoundError:
            logging.error("The file was not found. Please check the file path.")
            raise
        except pd.errors.EmptyDataError:
            logging.error("The file is empty. Please provide a valid CSV file.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while reading the data: {e}")
            raise

@step
def ingest_data() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
