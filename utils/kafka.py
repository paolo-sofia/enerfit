import json

from kafka3 import KafkaAdminClient, KafkaConsumer


def create_kafka_consumer(config: dict[str, str], encoding: str = "utf-8") -> KafkaConsumer:
    """Creates a Kafka consumer with the provided configuration.

    Args:
        config (dict): A dictionary containing the Kafka consumer configuration.
        encoding (str, optional): The encoding to use for deserializing key and value. Defaults to "utf-8".

    Returns:
        KafkaConsumer: The created Kafka consumer.

    Examples:
        >>> config = {
        ...     "bootstrap_servers": "localhost:9092",
        ...     "auto_offset_reset": "earliest",
        ...     "enable_auto_commit": True,
        ...     "group_id": "my-group"
        ... }
        >>> create_kafka_consumer(config)
        <kafka.consumer.KafkaConsumer object at 0x7f9a3e6e5a90>
    """
    return KafkaConsumer(
        bootstrap_servers=config.get("bootstrap_servers", ""),
        auto_offset_reset=config.get("auto_offset_reset", "earliest"),
        enable_auto_commit=config.get("enable_auto_commit", False),
        group_id=config.get("group_id", ""),
        key_deserializer=lambda x: json.loads(x.decode(encoding)),
        value_deserializer=lambda x: json.loads(x.decode(encoding)),
    )


def kafka_consumer_seek_to_last_committed_message(consumer: KafkaConsumer) -> KafkaConsumer:
    """Seeks the Kafka consumer to the last committed message for each partition.

    Args:
        consumer (KafkaConsumer): The Kafka consumer to seek.

    Returns:
        KafkaConsumer: The Kafka consumer after seeking to the last committed message.
    """
    partitions: set = consumer.assignment()

    for partition in partitions:
        if committed_offset := consumer.committed(partition):
            consumer.seek(partition, committed_offset.offset)
        else:
            consumer.seek_to_beginning(partition)
    return consumer


def commit_processed_messages_from_topic() -> None:
    """Commits processed messages from a Kafka topic for a specific day.

    Returns:
        None
    """
    consumer: KafkaConsumer = create_kafka_consumer({})
    consumer = kafka_consumer_seek_to_last_committed_message(consumer)

    for message in consumer:
        consumer.commit()


def create_consumer_and_seek_to_last_committed_message(
    config: dict[str, str], encoding: str = "utf-8"
) -> KafkaConsumer:
    """Creates a Kafka consumer with the provided configuration and seeks to the last committed message.

    Args:
        config (dict): A dictionary containing the Kafka consumer configuration.
        encoding (str, optional): The encoding used for deserializing messages. Defaults to "utf-8".

    Returns:
        KafkaConsumer: The created Kafka consumer.

    Examples:
        >>> config = {
        ...     "bootstrap_servers": "localhost:9092",
        ...     "auto_offset_reset": "earliest",
        ...     "enable_auto_commit": True,
        ...     "group_id": "my-group"
        ... }
        >>> create_consumer_and_seek_to_last_committed_message(config)
        <kafka.consumer.KafkaConsumer object at 0x7f9a3e6e5a90>
    """
    consumer: KafkaConsumer = create_kafka_consumer(config=config, encoding=encoding)
    kafka_consumer_seek_to_last_committed_message(consumer)
    return consumer


def delete_topic(kafka_config: dict[str, str], topic_name: str) -> bool:
    """Deletes a Kafka topic or a list of Kafka topics.

    Args:
        kafka_config (dict): A dictionary containing the Kafka configuration.
        topic_name (str or list): The name of the topic(s) to be deleted.

    Returns:
        bool: True if the topic(s) were successfully deleted, False otherwise.
    """
    if isinstance(topic_name, str):
        topic_name = [topic_name]
    if not isinstance(topic_name, list):
        raise TypeError(f"topic_name must be either a str or a list[str]. type given is {type(topic_name)}")

    try:
        admin_client = KafkaAdminClient(bootstrap_servers=kafka_config.get("bootstrap_servers", ""))

        admin_client.delete_topics(topics=[topic_name])

        admin_client.close()
    except Exception as e:
        print(e)
        return False

    return True


def get_all_topics(kafka_config: dict[str]) -> list[str]:
    """Retrieves a list of all Kafka topics.

    Args:
        kafka_config (dict): A dictionary containing the Kafka configuration.

    Returns:
        list: A list of all Kafka topics.

    Examples:
        >>> kafka_config = {
        ...     "bootstrap_servers": "localhost:9092"
        ... }
        >>> get_all_topics(kafka_config)
        ['topic1', 'topic2', 'topic3']
    """
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=kafka_config.get("bootstrap_servers", ""))
        topics: list[str] = admin_client.list_topics().topics.keys()
        # Close the AdminClient
        admin_client.close()
    except Exception as e:
        print(e)
        topics: list[str] = []

    return topics
