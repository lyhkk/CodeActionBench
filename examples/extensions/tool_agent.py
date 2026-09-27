"""Exercise an added tool and the world-frame tool through the public client."""


def run(context, client):
    if not client.call("capture_head", {})["images"]:
        raise RuntimeError("image tool result did not reach the agent")
    value = "caller-supplied text"
    if client.call("echo_value", {"value": value})["result"] != {"value": value}:
        raise RuntimeError("echo_value did not preserve the caller's argument")
    client.call("get_world_frame", {})
    client.call("done", {"report": "Local tool calls completed.", "success_claim": False})
