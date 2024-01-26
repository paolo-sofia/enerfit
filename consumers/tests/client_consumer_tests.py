import datetime
import pathlib
from unittest.mock import MagicMock

import pytest
import pytz

from consumers.client_consumer.main import main

# Constants for tests
CONFIG_PATH: pathlib.Path = pathlib.Path(__file__).parent / "config.toml"
YESTERDAY: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Tallinn")).date() - datetime.timedelta(days=1)


@pytest.mark.parametrize(
    "test_id, clients_data, is_empty, data_saved, expected_print, expected_update_args",
    [
        # Happy path tests
        ("happy-path-non-empty", [{"client_id": 1}], False, True, None, (True, YESTERDAY)),
        ("happy-path-empty", [], True, False, f"No data from {YESTERDAY}", (False, YESTERDAY)),
        # Edge cases
        ("edge-case-single-client", [{"client_id": 1}], False, True, None, (True, YESTERDAY)),
        ("edge-case-max-int-client-id", [{"client_id": 2**31 - 1}], False, True, None, (True, YESTERDAY)),
        # Error cases
        ("error-case-fetch-fail", None, False, False, "Error fetching clients", (False, YESTERDAY)),
        ("error-case-save-fail", [{"client_id": 1}], False, False, None, (False, YESTERDAY)),
    ],
)
def test_main(test_id, clients_data, is_empty, data_saved, expected_print, expected_update_args, monkeypatch):
    # Arrange
    config_mock: MagicMock = MagicMock(return_value={"some_key": "some_value"})
    monkeypatch.setattr("utils.configs.load_config", config_mock)
    clients_mock: MagicMock = MagicMock()
    clients_mock.is_empty.return_value = is_empty
    fetch_clients_mock: MagicMock = MagicMock(return_value=clients_mock)
    monkeypatch.setattr("main.fetch_clients_at_day", fetch_clients_mock)
    save_clients_mock: MagicMock = MagicMock(return_value=data_saved)
    monkeypatch.setattr("main.save_clients_to_datalake", save_clients_mock)
    update_processed_mock: MagicMock = MagicMock()
    monkeypatch.setattr("main.update_processed_clients_table", update_processed_mock)
    commit_processed_mock: MagicMock = MagicMock()
    monkeypatch.setattr("main.commit_processed_messages_from_topic", commit_processed_mock)
    print_mock: MagicMock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    # Act
    main()

    # Assert
    config_mock.assert_called_once_with(CONFIG_PATH)
    fetch_clients_mock.assert_called_once_with(config=config_mock.return_value, day=YESTERDAY)
    if not is_empty:
        save_clients_mock.assert_called_once_with(clients_mock, YESTERDAY)
        if data_saved:
            commit_processed_mock.assert_called_once_with(YESTERDAY)
    update_processed_mock.assert_called_once_with(*expected_update_args)
    if expected_print:
        print_mock.assert_called_once_with(expected_print)
