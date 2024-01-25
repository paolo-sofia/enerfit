import pathlib
from threading import Thread

from producers.generic_producer import GenericProducer


def main() -> None:
    """Runs the main function to start the producers.

    Returns:
        None
    """
    producers: list[GenericProducer] = [
        GenericProducer(config_file=path) for path in (pathlib.Path(__file__).parent / "configs").rglob("*.toml")
    ]
    threads: list[Thread] = []
    for producer in producers:
        producer.__post_init__()
        producer_thread: Thread = Thread(target=producer.producer_loop)
        threads.append(producer_thread)

    for sensor_thread in threads:
        print(f"starting {sensor_thread.name}")
        sensor_thread.start()


if __name__ == "__main__":
    main()
