# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime

import pandas as pd
import pytz


def preprocess_clients_columns(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def set_column_types(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def fetch_clients_at_day(day: datetime.date) -> pd.DataFrame:
    return pd.DataFrame()


def save_clients_to_datalake(data: pd.DataFrame, day: datetime.date) -> bool:
    return False


if __name__ == "__main__":
    yesterday: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Tallin")).date() - datetime.timedelta(
        days=1
    )

    ## Prenditi tutti i dati di ieri da kafka
    clients: pd.DataFrame = fetch_clients_at_day(day=yesterday)

    clients = set_column_types(clients)

    clients = preprocess_clients_columns(clients)

    save_clients_to_datalake(
        clients,
    )
