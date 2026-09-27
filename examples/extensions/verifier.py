"""The environment's current predicate, exposed through the local verifier interface."""


def verify(spec, *, env, **context):
    observed = env.check_success()
    if observed is None:
        raise ValueError("environment success evidence is unavailable")
    success = bool(observed)
    return {"success": success, "score": 1.0 if success else 0.0,
            "kind": "local_success", "detail": {"check_success_result": success}}
