"""Pure contact-feature extraction shared by the live monitor and the calibration probe.

Why this is its own module rather than probe-local code: the interruption thresholds have to be
chosen from measured distributions, and a distribution measured with a different pair keying or a
different impulse aggregation than the rule that later enforces it is not evidence about that rule.
``CollisionMonitor`` therefore keys contacts through ``pair_key`` here, and the probe's per-step
aggregates come from ``contact_features``; their agreement is pinned by a unit test rather than by
a comment.

This module makes NO interruption decision and holds no threshold. Severity rules are deliberately
absent until the calibration data exists -- what it produces is the raw per-step time series a
parameter sweep can be run over offline, so one expensive simulator campaign can answer many
candidate rules instead of one.

No simulator import: every entry point takes already-read contact objects, so the whole module is
testable on a laptop. Entity names appear in its output because it is a HOST-ONLY instrument; the
model-facing path keeps using the monitor's opaque keys.
"""
from collections import deque


def entity_name(entity):
    getter = getattr(entity, "get_name", None)
    if callable(getter):
        return str(getter())
    return str(getattr(entity, "name", ""))


def pair_key(left_name, right_name, robot_names):
    """Canonical key for one contacting entity pair, or ``None`` when no robot link is involved.

    Robot-to-robot keys sort their two link names so the same self-contact cannot appear under two
    keys depending on which body the solver listed first. Robot-to-world keys put the robot link
    second so the key shape alone says which side is the robot.
    """
    left_robot = left_name in robot_names
    right_robot = right_name in robot_names
    if not left_robot and not right_robot:
        return None
    if left_robot and right_robot:
        key = ("robot", *sorted((left_name, right_name)))
        return key, "robot_robot", list(key[1:]), None
    if left_robot:
        return ("world", left_name, right_name), "robot_world", [left_name], right_name
    return ("world", right_name, left_name), "robot_world", [right_name], left_name


def _point_impulse_vectors(contact, max_points):
    vectors = []
    for point in list(getattr(contact, "points", ()) or ())[:max_points]:
        raw = getattr(point, "impulse", None)
        try:
            vectors.append([float(value) for value in raw][:3])
        except (TypeError, ValueError):
            continue
    return vectors


def _norm(vector):
    return sum(value ** 2 for value in vector) ** 0.5


def contact_features(contacts, robot_names, *, finger_names=None, max_points=32):
    """Per-pair features for ONE physics step, with no threshold applied.

    Two impulse aggregations are recorded on purpose, because which one separates the classes
    better is itself an open question: ``j_sum_ns`` sums the per-point magnitudes (conservative,
    but it grows with how many points the solver happened to generate) and ``j_net_ns`` is the
    magnitude of the summed vector (steadier against point count, but opposing points cancel).
    Freezing one before looking at the data would be choosing the answer in advance.
    """
    fingers = set()
    for arm in ("left", "right"):
        fingers.update((finger_names or {}).get(arm, ()))
    out = {}
    for contact in contacts or ():
        try:
            left, right = contact.bodies[0].entity, contact.bodies[1].entity
        except Exception:
            continue
        resolved = pair_key(entity_name(left), entity_name(right), robot_names)
        if resolved is None:
            continue
        key, category, robot_links, world_entity = resolved
        vectors = _point_impulse_vectors(contact, max_points)
        record = out.setdefault(key, {
            "category": category,
            "robot_links": robot_links,
            "world_entity": world_entity,
            "finger_only": all(link in fingers for link in robot_links) and bool(fingers),
            "j_sum_ns": 0.0,
            "j_net_ns": 0.0,
            "max_point_ns": 0.0,
            "point_count": 0,
            "_net": [0.0, 0.0, 0.0],
        })
        for vector in vectors:
            magnitude = _norm(vector)
            record["j_sum_ns"] += magnitude
            record["max_point_ns"] = max(record["max_point_ns"], magnitude)
            for axis in range(min(3, len(vector))):
                record["_net"][axis] += vector[axis]
        record["point_count"] += len(vectors)
    # A world entity that is touching a gripper's fingers in this same step, AND also touching some
    # other robot link, is one physical situation the flat robot/world split cannot express: an
    # object the robot is holding, brushing the robot itself. Measured in the shadow campaign as
    # four of the five non-finger robot-to-world events (`020_hammer` against `left_camera`,
    # `right_camera`, `fl_link6`, `fr_link6` while the hammer was being carried), where reading
    # them as environment collisions is simply wrong.
    #
    # Derived only from the contact graph, never from scene ground truth, and named for what is
    # actually observed: this says the entity is in finger contact, NOT that a grasp is holding it.
    # An object merely resting against a fingertip satisfies it too.
    finger_touched = {
        record["world_entity"] for record in out.values()
        if record["world_entity"] is not None and record["finger_only"]}
    for record in out.values():
        record["j_net_ns"] = _norm(record.pop("_net"))
        record["world_entity_in_finger_contact"] = (
            record["world_entity"] in finger_touched if record["world_entity"] else False)
    return out


def contact_class(record, *, finger_names=None):
    """Name the part decomposition a rule can act on. Descriptive; decides nothing.

    Measured over the shadow campaign, only ``same_gripper_fingers`` is a class whose members were
    all benign. Every other class held both legitimate task contact and harmful contact -- the
    open_laptop expert rests its own wrist on the table at 0.899 N.s, the same contact a live run
    was interrupted for three times -- so part alone cannot decide, and these names exist to be
    crossed with the acting action's declared intent.
    """
    fingers = set()
    for arm in ("left", "right"):
        fingers.update((finger_names or {}).get(arm, ()))
    links = record["robot_links"]
    if record["category"] == "robot_robot":
        same_arm = len({link.split("_")[0] for link in links}) == 1
        if fingers and all(link in fingers for link in links) and same_arm:
            return "same_gripper_fingers"
        return "robot_self_or_cross_arm"
    if record.get("world_entity_in_finger_contact") and not record["finger_only"]:
        return "finger_held_entity_against_robot"
    return "fingertip_world" if record["finger_only"] else "non_finger_world"


class ContactEventTracker:
    """Assemble per-step features into per-pair contact events. Records; never interrupts.

    One event is one continuous stay of a pair, tolerating short solver chatter (``close_after``
    absent steps). It keeps a bounded head of the per-step series so an offline sweep can ask
    "would a rule with window W and floor J have fired, and when" without re-running the simulator,
    plus unbounded running aggregates so a long press is still summarized after the head is full.
    """

    def __init__(self, *, series_cap=500, close_after=5):
        self.series_cap = int(series_cap)
        self.close_after = int(close_after)
        self._open = {}
        self._closed = []

    def observe(self, step, features, *, baseline_keys=frozenset(), context=None):
        step = int(step)
        # Retire stale stays BEFORE recording this step, or a pair that returns after a long
        # absence would extend the old event instead of starting a new one -- and "one press" and
        # "two presses separated by a retreat" are different facts for a duration rule.
        for key in [k for k, event in self._open.items()
                    if step - event["last_step"] > self.close_after]:
            self._closed.append(self._open.pop(key))
        for key, record in features.items():
            event = self._open.get(key)
            if event is None:
                event = {
                    "key": list(key),
                    "category": record["category"],
                    "robot_links": list(record["robot_links"]),
                    "world_entity": record["world_entity"],
                    "finger_only": bool(record["finger_only"]),
                    "in_baseline": key in baseline_keys,
                    "context": dict(context or {}),
                    "first_step": step,
                    "last_step": step,
                    "steps_present": 0,
                    "j_sum_max": 0.0,
                    "j_net_max": 0.0,
                    "j_sum_total": 0.0,
                    "series": [],
                    "series_truncated": False,
                }
                self._open[key] = event
            event["last_step"] = step
            event["steps_present"] += 1
            event["j_sum_max"] = max(event["j_sum_max"], record["j_sum_ns"])
            event["j_net_max"] = max(event["j_net_max"], record["j_net_ns"])
            event["j_sum_total"] += record["j_sum_ns"]
            if len(event["series"]) < self.series_cap:
                event["series"].append(
                    [step, round(record["j_sum_ns"], 12), round(record["j_net_ns"], 12),
                     record["point_count"]])
            else:
                event["series_truncated"] = True

    def close_all(self):
        for key in list(self._open):
            self._closed.append(self._open.pop(key))

    def events(self):
        """Closed events plus a snapshot of the still-open ones, ordered by onset."""
        return sorted(self._closed + list(self._open.values()),
                      key=lambda event: (event["first_step"], event["key"]))


class SampledSeries:
    """A bounded ring of periodic host-side samples (joint velocity, TCP, whatever the caller reads).

    The point is to keep the 250 Hz callback cheap: approach speed at contact onset is recoverable
    offline from a 50 Hz sample, and paying for a full state read every step would change the very
    timing the campaign is measuring.
    """

    def __init__(self, *, every=5, capacity=4000):
        self.every = max(1, int(every))
        self._samples = deque(maxlen=int(capacity))

    def maybe_sample(self, step, read):
        if int(step) % self.every:
            return
        try:
            value = read()
        except Exception:
            return
        self._samples.append([int(step), value])

    def samples(self):
        return [list(item) for item in self._samples]


__all__ = ["ContactEventTracker", "SampledSeries", "contact_class", "contact_features",
           "entity_name", "pair_key"]
