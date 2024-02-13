def check_model_type(model_type: str) -> str | None:
    """Check and validate the given model type.

    Args:
        model_type: The type of the model.

    Returns:
        str | None: The validated model type if it is either 'producer' or 'consumer', None otherwise.

    Raises:
        ValueError: If the model type is not 'producer' or 'consumer'.
    """
    model_type = model_type.lower()

    if model_type not in ["producer", "consumer"]:
        raise ValueError(f"Model type must be either 'producer' or 'consumer', given model type is: {model_type}")

    return model_type
