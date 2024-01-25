from unittest.mock import MagicMock, Mock, patch

import pytest

from producers.main import main


@pytest.fixture
def mock_producer() -> MagicMock:
    """Fixture for mocking the GenericProducer class.

    Returns:
        Mock: The mocked GenericProducer class.
    """
    with patch("main.GenericProducer") as mock:
        yield mock


@pytest.fixture
def mock_thread() -> MagicMock:
    """Fixture for mocking the Thread class.

    Returns:
        Mock: The mocked Thread class.
    """
    with patch("main.Thread") as mock:
        yield mock


def test_main(mock_producer: MagicMock, mock_thread: MagicMock) -> None:
    client_config_file_path: str = "tests/data/configs/client.toml"
    energy_config_file_path: str = "tests/data/configs/energy.toml"

    # Mock the producer instances
    mock_producers: list[Mock] = [Mock(), Mock()]
    mock_producer.side_effect = mock_producers

    # Mock the thread instances
    mock_threads: list[Mock] = [Mock(), Mock()]
    mock_thread.side_effect = mock_threads

    # Call the main function
    main()

    # Verify that GenericProducer instances are created with the correct paths
    mock_producer.assert_has_calls(
        [
            pytest.call(config_file=client_config_file_path),
            pytest.call(config_file=energy_config_file_path),
        ],
        any_order=True,
    )

    # Verify that Thread instances are created for each GenericProducer
    mock_thread.assert_has_calls(
        [
            pytest.call(target=mock_producers[0].producer_loop),
            pytest.call(target=mock_producers[1].producer_loop),
        ],
        any_order=True,
    )

    # Verify that start() is called on each Thread instance
    for thread in mock_threads:
        thread.start.assert_called_once()
