"""Credential-free example of a Python agent using the public image/tool client."""


def run(context, client):
    observation = client.call("capture_head", {})
    if not observation["images"]:
        raise RuntimeError("image tool result did not reach the agent")
    client.call("get_robot_state", {"arms": ["left", "right"]})
    client.call("done", {"report": "Local agent lifecycle check completed.", "success_claim": False})
