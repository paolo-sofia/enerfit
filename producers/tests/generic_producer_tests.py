import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import tomllib

from producers.generic_producer import GenericProducer

# Constants for tests
TEST_CONFIG_FILE = Path("/path/to/test/config.toml")
TEST_INPUT_DATA_PATH = Path("/path/to/test/data.csv")
TEST_TOPIC = "test_topic"
TEST_KEY = "test_key"
TEST_NAME = "test_producer"
TEST_FREQUENCY = 2.0
ENCODING = "utf-8"

# Sample data for tests
SAMPLE_CONFIG = {
    "producer": {"bootstrap_servers": "localhost:9092", "api_version": (0, 10, 1)},
    "info": {"key": TEST_KEY, "name": TEST_NAME, "topic": TEST_TOPIC, "frequency": TEST_FREQUENCY},
}
SAMPLE_DATA = pd.DataFrame({"sensor_id": [1, 2], "temperature": [22.5, 23.0], "humidity": [45, 50]})


@pytest.fixture
def mock_kafka_producer():
    with patch("generic_producer.KafkaProducer") as mock:
        yield mock


@pytest.fixture
def mock_toml_load():
    with patch("generic_producer.tomllib.load", return_value=SAMPLE_CONFIG) as mock:
        yield mock


@pytest.fixture
def mock_sleep():
    with patch("generic_producer.sleep") as mock:
        yield mock


@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv", return_value=SAMPLE_DATA) as mock:
        yield mock


@pytest.mark.parametrize(
    "config_file, input_data_path, frequency, expected_frequency, test_id",
    [
        (TEST_CONFIG_FILE, TEST_INPUT_DATA_PATH, 1.0, 1.0, "happy_path_default_frequency"),
        (TEST_CONFIG_FILE, TEST_INPUT_DATA_PATH, 0.5, 0.5, "happy_path_custom_frequency"),
    ],
)
def test_generic_producer_initialization(
    mock_kafka_producer, mock_toml_load, config_file, input_data_path, frequency, expected_frequency, test_id
):
    producer: GenericProducer = GenericProducer(
        config_file=config_file, input_data_path=input_data_path, frequency=frequency
    )
    producer.__post_init__()

    mock_toml_load.assert_called_once_with(config_file.open("rb"))
    mock_kafka_producer.assert_called_once()
    assert producer.key == TEST_KEY
    assert producer.name == TEST_NAME
    assert producer.topic == TEST_TOPIC
    assert producer.frequency == expected_frequency


@pytest.mark.parametrize(
    "row, test_id",
    [
        (SAMPLE_DATA.iloc[0], "happy_path_first_row"),
        (SAMPLE_DATA.iloc[1], "happy_path_second_row"),
    ],
)
def test_publish_message(mock_kafka_producer, mock_read_csv, row, test_id):
    # Arrange
    producer = GenericProducer(config_file=TEST_CONFIG_FILE, input_data_path=TEST_INPUT_DATA_PATH)
    producer.__post_init__()

    # Act
    producer.publish_message(row)

    # Assert
    mock_kafka_producer.return_value.send.assert_called_once()
    args, kwargs = mock_kafka_producer.return_value.send.call_args
    assert kwargs["topic"] == TEST_TOPIC
    assert TEST_KEY.encode(ENCODING) == kwargs["key"]
    assert json.loads(kwargs["value"].decode(ENCODING))["value"] == row.to_dict()


@pytest.mark.parametrize(
    "input_data_path, frequency, test_id",
    [
        (TEST_INPUT_DATA_PATH, 2.0, "happy_path_producer_loop"),
        (TEST_INPUT_DATA_PATH, 0.1, "edge_case_high_frequency"),
    ],
)
def test_producer_loop(mock_kafka_producer, mock_read_csv, mock_sleep, input_data_path, frequency, test_id):
    # Arrange
    producer = GenericProducer(config_file=TEST_CONFIG_FILE, input_data_path=input_data_path, frequency=frequency)
    producer.__post_init__()

    # Act
    producer.producer_loop()

    # Assert
    assert mock_read_csv.call_count == 1
    assert mock_kafka_producer.return_value.send.call_count == len(SAMPLE_DATA)
    assert mock_sleep.call_count == len(SAMPLE_DATA)
    mock_sleep.assert_called_with(1 / frequency)


@pytest.mark.parametrize(
    "input_data_path, test_id",
    [
        (None, "error_case_no_input_data_path"),
    ],
)
def test_producer_loop_error(mock_kafka_producer, input_data_path, test_id):
    # Arrange
    producer = GenericProducer(config_file=TEST_CONFIG_FILE, input_data_path=input_data_path)

    # Act / Assert
    with pytest.raises(AttributeError):
        producer.producer_loop()


@pytest.mark.parametrize(
    "config_file, test_id",
    [
        (Path("/path/to/invalid/config.toml"), "error_case_invalid_config"),
    ],
)
def test_load_config_error(mock_toml_load, config_file, test_id):
    # Arrange
    mock_toml_load.side_effect = tomllib.TOMLDecodeError("Invalid TOML file", 0, 0)
    producer = GenericProducer(config_file=config_file)

    # Act / Assert
    with pytest.raises(tomllib.TOMLDecodeError):
        producer.load_config(config_file)
