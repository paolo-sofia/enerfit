import pathlib
from threading import Thread

from producers.generic_producer import GenericProducer

if __name__ == "__main__":
    producers: list[GenericProducer] = [
        GenericProducer(config_file=path) for path in (pathlib.Path(__file__) / "configs").rglob("*.toml")
    ]
    threads: list[Thread] = []
    for producer in producers:
        producer_thread: Thread = Thread(target=producer.producer_loop)
        threads.append(producer_thread)

    for sensor_thread in threads:
        sensor_thread.start()
