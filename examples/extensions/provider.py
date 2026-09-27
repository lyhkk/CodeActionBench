"""Provider factory example using the existing OpenAI-compatible implementation."""


def create(**options):
    from codeaction.providers.model_adapter import ModelAdapter
    if not options.get("base_url"):
        raise ValueError("local_provider requires the credential alias's BASE_URL")
    return ModelAdapter(**options)
