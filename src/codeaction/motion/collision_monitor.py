"""Read robot contact facts at motion boundaries without classifying task intent.

Contact identity never changes execution and contacted world entities never cross the model
boundary.  Waypoint primitives decide whether they stalled from measured motion progress, then
ask this reader whether contact coexisted with that stop.  It reports facts only.
"""
import copy

from codeaction.motion.action_contact_policy import is_same_gripper_finger_pair
from codeaction.motion.contact_features import entity_name as _entity_name_shared, pair_key
from codeaction.contracts.failures import ContactReadUnavailable

# There is deliberately no `MOTION_TOOLS` alias here. `action_contact_policy.PHYSICS_ACTION_TOOLS`
# is the live physics surface; the analysis sets in `codeaction/metrics.py` and `data/make_reports.py`
# are DIFFERENT sets on purpose (they retain retired tool names so archived transcripts keep their
# meaning, and they exclude actions with no Cartesian displacement to attribute). The relationship
# between them is pinned by a test rather than collapsed into one shared object.


# One naming rule for the evidence reader and calibration probes, for the same reason as `pair_key`.
_entity_name = _entity_name_shared


def _vector(value):
    """Convert a simulator vector to bounded JSON-safe floats without assuming its concrete type."""
    if value is None:
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    try:
        return [float(item) for item in values[:4]]
    except (TypeError, ValueError):
        return None


def _scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_scene_contacts(env):
    """Read the scene contact list without turning backend failure into a negative measurement."""
    try:
        return tuple(env.scene.get_contacts() or ())
    except Exception as exc:
        raise ContactReadUnavailable() from exc


class CollisionMonitor:
    """Read current robot contact while withholding contacted world identity."""

    def __init__(self, env, impulse_threshold):
        self._env = env
        self._threshold = float(impulse_threshold)
        self._robot_names, self._finger_names = self._read_robot_links()

    def _read_robot_links(self):
        robot = getattr(self._env, "robot", None)
        robot_links = set()
        # Kept per arm as well: a report that says "something other than your fingertips, on the
        # LEFT arm" is actionable, while the raw URDF link name is a token the tool surface never
        # taught the agent to interpret.
        self._arm_links = {"left": set(), "right": set()}
        for attr, arm in (("left_entity", "left"), ("right_entity", "right")):
            entity = getattr(robot, attr, None)
            links = getattr(entity, "get_links", None)
            links = links() if callable(links) else getattr(entity, "links", ())
            for link in links or ():
                name = _entity_name(link)
                robot_links.add(name)
                self._arm_links[arm].add(name)

        finger_links = {"left": set(), "right": set()}
        for arm in ("left", "right"):
            for item in getattr(robot, f"{arm}_gripper", ()) or ():
                try:
                    finger_links[arm].add(_entity_name(item[0].child_link))
                except Exception:
                    continue
        return robot_links, finger_links

    def _contacts(self, *, detailed=False, apply_impulse_threshold=True):
        """Return current robot contacts keyed by the historical pair signature.

        Normal stall reporting keeps only keys. Probe-only callers may request geometry without
        changing execution.
        """
        contacts = read_scene_contacts(self._env)
        records = {}
        for contact in contacts or ():
            try:
                left, right = contact.bodies[0].entity, contact.bodies[1].entity
            except Exception:
                continue
            left_name, right_name = _entity_name(left), _entity_name(right)
            left_robot = left_name in self._robot_names
            right_robot = right_name in self._robot_names
            if not left_robot and not right_robot:
                continue
            point_records = []
            impulse = 0.0
            for point in list(getattr(contact, "points", ()) or ())[:32]:
                raw_impulse = getattr(point, "impulse", None)
                try:
                    impulse_norm = sum(float(value) ** 2 for value in raw_impulse) ** 0.5
                except (TypeError, ValueError):
                    impulse_norm = 0.0
                impulse += impulse_norm
                if detailed:
                    point_records.append({
                        "impulse_ns": _vector(raw_impulse),
                        "impulse_norm_ns": impulse_norm,
                        "position_world_m": _vector(getattr(point, "position", None)),
                        "normal_world": _vector(getattr(point, "normal", None)),
                        "separation_m": _scalar(getattr(point, "separation", None)),
                    })
            if apply_impulse_threshold and impulse <= self._threshold:
                continue
            # Keyed through the shared pure helper so probes and boundary reports group contacts
            # identically.
            key, category, robot_links, world_entity = pair_key(
                left_name, right_name, self._robot_names)
            # One gripper closing onto itself is not a contact between the robot and anything;
            # dropped from BOTH the baseline and the live set so the two stay comparable.
            if is_same_gripper_finger_pair(key, self._finger_names):
                continue
            if not detailed:
                records.setdefault(key, None)
                continue
            record = records.setdefault(key, {
                "category": category,
                "robot_links": robot_links,
                "world_entity": world_entity,
                "sum_impulse_ns": 0.0,
                "max_point_impulse_ns": 0.0,
                "point_count": 0,
                "points": [],
            })
            record["sum_impulse_ns"] += impulse
            record["max_point_impulse_ns"] = max(
                record["max_point_impulse_ns"],
                max((point["impulse_norm_ns"] for point in point_records), default=0.0))
            record["point_count"] += len(point_records)
            record["points"].extend(point_records)
        return records

    def attach_pose_reader(self, fn):
        """Attach the host's own robot-state read, sampled only at a reported stall."""
        self._pose_reader = fn if callable(fn) else None

    def _arm_of(self, link):
        for arm in ("left", "right"):
            if link in getattr(self, "_arm_links", {}).get(arm, ()):
                return arm
        return None

    def _parts(self, keys):
        """Name the robot's own contacting parts. No world entity identity, by construction."""
        fingers = set()
        for arm in ("left", "right"):
            fingers.update(self._finger_names.get(arm, ()))
        out = []
        for key in sorted(keys):
            links = list(key[1:]) if key[0] == "robot" else [key[1]]
            arms = sorted({arm for arm in (self._arm_of(link) for link in links) if arm})
            if key[0] == "robot":
                part = "robot_self"
            elif all(link in fingers for link in links):
                part = "fingertip"
            else:
                part = "arm_link"
            out.append({"part": part, "arms": arms})
        return out

    def current_contact_evidence(self, arm):
        """Return agent-safe contact facts at one caller-selected motion boundary.

        A waypoint loop calls this only after it has independently decided that it stalled. The
        result reports contact that coexisted with that stop; it does not claim the contact caused
        the stall. The contacted entity identity never leaves ``_contacts``. An unavailable read
        is ``None``, not a false no-contact claim.
        """
        selected_arm = str(arm)
        if selected_arm not in ("left", "right"):
            raise ValueError("current_contact_evidence arm must be 'left' or 'right'")
        try:
            keys = set(self._contacts())
        except ContactReadUnavailable:
            return {"blocked_in_contact": None,
                    "contact_parts": [], "pose_at_contact": None}
        selected_links = self._arm_links[selected_arm]
        keys = {
            key for key in keys
            if any(link in selected_links
                   for link in (key[1:] if key[0] == "robot" else (key[1],)))
        }
        if not keys:
            return {"blocked_in_contact": False,
                    "contact_parts": [], "pose_at_contact": None}
        pose = None
        reader = getattr(self, "_pose_reader", None)
        if callable(reader):
            try:
                pose = reader()
            except Exception:
                pose = None
        return {
            "blocked_in_contact": True,
            "contact_parts": self._parts(keys),
            "pose_at_contact": copy.deepcopy(pose) if isinstance(pose, dict) else None,
        }


__all__ = ["CollisionMonitor", "read_scene_contacts"]
