import json
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Self

import pandas as pd
import tomllib
from kafka3 import KafkaProducer
from kafka3.producer.future import FutureRecordMetadata

ENCODING: str = "utf-8"


@dataclass
class GenericProducer:
    """Abstract base class for producers.

    Args:
        config_file: The path to the configuration file.
        key: The key of the producer.
        name: The name of the producer.
        topic: The topic to publish messages to.
        input_data_path: The path to the input data file.
        frequency: The frequency at which to publish messages.
        producer: The KafkaProducer instance.

    Methods:
        publish_message: Publishes a message by sending the data readings to a specified topic.
        publish_producer_initialization: Publishes the producer initialization message to a specified topic.
        producer_loop: Runs the main loop for this producer.

    Returns:
        None
    """

    config_file: pathlib.Path = field(default_factory=Path)
    key: str = ""
    name: str = ""
    topic: str = ""
    input_data_path: pathlib.Path = field(default_factory=Path)
    frequency: float = 1.0
    producer: KafkaProducer = None

    def __post_init__(self: Self) -> None:
        """Performs initialization tasks after object creation.

        Args:
            self: The instance of the class.

        Returns:
            None
        """
        self.load_config(self.config_file)

    def load_config(self: Self, config_file: pathlib.Path) -> None:
        """Loads the configuration from a specified file and initializes the producer.

        Args:
            self: The instance of the class.
            config_file: The path to the configuration file.

        Returns:
            None
        """
        with config_file.open("rb") as f:
            config: dict[str, str | dict[str, str | tuple]] = tomllib.load(f)

        config["producer"]["api_version"] = tuple(config["producer"]["api_version"])
        self.producer: KafkaProducer = KafkaProducer(**dict(config["producer"]))
        self.key: str = config.get("info", {}).get("key", "")
        self.name: str = config.get("info", {}).get("name", "")
        self.topic: str = config.get("info", {}).get("topic", "")
        self.frequency: float = float(config.get("info", {}).get("frequency", 1.0))

    def publish_message(self: Self, row: pd.Series) -> None:
        """Publishes a message by sending the data readings to a specified topic.

        Args:
            self: The instance of the class.
            row: A pandas Series representing the data readings.

        Returns:
            None
        """
        data_readings: dict[str, str | pd.Series] = {"key": self.id, "name": self.name, "value": row.to_dict()}
        data_readings: bytes = json.dumps(data_readings).encode(ENCODING)

        return_val: FutureRecordMetadata = self.producer.send(
            topic=self.topic, value=data_readings, key=self.id.encode(ENCODING)
        )
        print(return_val)

    def publish_producer_initialization(self: Self) -> None:
        """Publishes the producer initialization message to a specified topic.

        Args:
            self: The instance of the class.

        Returns:
            None
        """
        message: dict[str, str] = {"key": self.key, "name": self.name}
        message: bytes = json.dumps(message).encode(ENCODING)
        self.producer.send(self.topic, value=message, key=self.key.encode(ENCODING))

    ##
    def producer_loop(self: Self) -> None:
        """Runs the main loop for this producer.

        a.k.a. reads a row from the input file at the specified interval and publish the value on the current topic.

        Args:
            self: The instance of the producer.

        Returns:
            None
        """
        if not self.input_data_path:
            raise AttributeError("input_data_path must be set")

        self.publish_info_initialization()
        dataframe: pd.DataFrame = pd.read_csv(self.input_data_path)
        for _, row in dataframe.iterrows():
            self.publish_message(row)
            sleep(1 / self.frequency)
