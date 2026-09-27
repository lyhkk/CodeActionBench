"""Gripper-to-object contact, for environment-owned success predicates.

An `is_*_gripper_close()` term is a JOINT-STATE test: it is equally true when the gripper closed on
the object and when it closed on air. Any predicate that means "this arm is holding that object"
therefore needs a contact term as well, and that term must name BOTH sides — the object and the
links of one specific arm — or it degenerates back into "something touched something".

Host-side env code: it reads the live contact set (ground truth) and is never importable from the
agent-facing surface. Kept object- and arm-agnostic so every envs_ext predicate uses one
implementation rather than a per-task copy.
"""


def gripper_link_names(env, arm: str) -> set:
    """The link names that make up one gripper: the fixed gripper body plus every driven finger."""
    arm = str(arm)
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    names = set(getattr(env.robot, f"{arm}_fix_gripper_name"))
    for joint, _multiplier, _offset in getattr(env.robot, f"{arm}_gripper"):
        if joint is not None:
            names.add(joint.child_link.get_name())
    return names


def gripper_contacts_actor(env, arm: str, actor) -> bool:
    """True when any link of `arm`'s gripper is in contact with `actor` right now.

    Contacts with no contact points are skipped: SAPIEN reports proximity pairs whose impulse set
    is empty, and counting those would make the term true just before the fingers close.
    """
    links = gripper_link_names(env, arm)
    targets = {actor.get_name()}
    targets.update(str(name) for name in getattr(actor, "link_dict", {}))
    for contact in env.scene.get_contacts():
        names = (contact.bodies[0].entity.name, contact.bodies[1].entity.name)
        target = next((name for name in names if name in targets), None)
        if not contact.points or target is None:
            continue
        other = names[1] if names[0] == target else names[0]
        if other in links:
            return True
    return False
