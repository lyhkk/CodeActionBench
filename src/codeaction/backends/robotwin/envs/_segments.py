"""Host-side reference segment library: reusable choreography EXTRACTED
from upstream play_once bodies (handover_block / open_microwave / stack_bowls_two patterns),
composed by envs_ext play_once implementations and reachability probes.

Never nests a full upstream play_once() — the audited reasons (check_success_audit categories):
sticky success flags, play_once-only attrs, and mid-episode self.check_success() calls
(open_microwave's expert loop does exactly that; the extraction below drives the SAME door-follow
loop on joint progress alone). Segments call only the generic Base_Task Action machinery
(grasp_actor / place_actor / move_by_displacement / open_gripper / back_to_origin) and read GT
freely — this package is host-side env code, never agent-visible.

Contract: every segment returns {"ok": bool, "detail": {...}}, never calls env.check_success()
(success judgment belongs to the out-of-band verifier alone), and respects env.plan_success as the
motion-failure signal exactly like upstream expert code. A segment entered with plan_success
already False refuses to act, so per-segment attribution stays clean."""
import numpy as np

from envs.utils.action import ArmTag


def _entry(env):
    if not bool(getattr(env, "plan_success", True)):
        return {"ok": False, "detail": {"skipped": "plan_success already False at entry"}}
    return None


def _result(env, **extra):
    ok = bool(env.plan_success)
    return {"ok": ok, "detail": {"plan_success": ok, **extra}}


def grasp(env, actor, arm, *, pre_grasp_dis=0.1, grasp_dis=0.0, contact_point_id=None,
          lift_z=0.1, retreat_other_arm=None):
    """Grasp an actor with one arm, optionally retreating the other arm in the same move.

    The concurrent retreat matches upstream dual-arm references: when objects are close, moving
    the previous arm home before securing the next object can sweep that unsecured object away.
    """
    bad = _entry(env)
    if bad:
        return bad
    kw = dict(arm_tag=ArmTag(str(arm)), pre_grasp_dis=float(pre_grasp_dis),
              grasp_dis=float(grasp_dis))
    if contact_point_id is not None:
        kw["contact_point_id"] = contact_point_id
    actions = [env.grasp_actor(actor, **kw)]
    if retreat_other_arm is not None:
        actions.append(env.back_to_origin(arm_tag=ArmTag(str(retreat_other_arm))))
    env.move(*actions)
    if lift_z and env.plan_success:
        env.move(env.move_by_displacement(ArmTag(str(arm)), z=float(lift_z)))
    return _result(env)


def dual_grasp(env, actor, *, left_contact_point_id, right_contact_point_id,
               pre_grasp_dis=0.035, grasp_dis=0.0, preclose=0.5):
    """Approach and grasp one shared payload at two declared contact points simultaneously."""
    bad = _entry(env)
    if bad:
        return bad
    env.move(env.close_gripper(ArmTag("left"), pos=float(preclose)),
             env.close_gripper(ArmTag("right"), pos=float(preclose)))
    if env.plan_success:
        env.move(
            env.grasp_actor(actor, arm_tag=ArmTag("left"),
                            pre_grasp_dis=float(pre_grasp_dis),
                            grasp_dis=float(grasp_dis),
                            contact_point_id=left_contact_point_id),
            env.grasp_actor(actor, arm_tag=ArmTag("right"),
                            pre_grasp_dis=float(pre_grasp_dis),
                            grasp_dis=float(grasp_dis),
                            contact_point_id=right_contact_point_id),
        )
    return _result(env)


def dual_displace(env, *, x=0.0, y=0.0, z=0.0, max_leg_m=0.08):
    """Translate both end effectors together using bounded world-frame legs."""
    bad = _entry(env)
    if bad:
        return bad
    delta = np.array([float(x), float(y), float(z)], dtype=float)
    legs = max(1, int(np.ceil(np.linalg.norm(delta) / float(max_leg_m))))
    step = delta / legs
    completed = 0
    for _ in range(legs):
        env.move(env.move_by_displacement(ArmTag("left"), x=step[0], y=step[1], z=step[2]),
                 env.move_by_displacement(ArmTag("right"), x=step[0], y=step[1], z=step[2]))
        if not env.plan_success:
            break
        completed += 1
    return _result(env, legs=legs, completed_legs=completed,
                   displacement_m=round(float(np.linalg.norm(delta)), 4))


def dual_release(env, *, retreat_z=0.0):
    """Open both grippers together, then optionally retreat both arms vertically."""
    bad = _entry(env)
    if bad:
        return bad
    env.move(env.open_gripper(ArmTag("left")), env.open_gripper(ArmTag("right")))
    if retreat_z and env.plan_success:
        env.move(env.move_by_displacement(ArmTag("left"), z=float(retreat_z)),
                 env.move_by_displacement(ArmTag("right"), z=float(retreat_z)))
    return _result(env)


def place(env, actor, arm, *, target_pose, functional_point_id=0, pre_dis=0.09, dis=0.0,
          is_open=True, constrain="free"):
    """Place a held actor at target_pose (xyz+quat list, or an env-provided pose)."""
    bad = _entry(env)
    if bad:
        return bad
    env.move(env.place_actor(actor, target_pose=list(target_pose), arm_tag=ArmTag(str(arm)),
                             functional_point_id=functional_point_id, pre_dis=float(pre_dis),
                             dis=float(dis), is_open=bool(is_open), constrain=constrain))
    return _result(env)


def retreat_home(env, arm):
    """Return one arm to its origin posture (upstream back_to_origin), clearing the workspace."""
    bad = _entry(env)
    if bad:
        return bad
    env.move(env.back_to_origin(arm_tag=ArmTag(str(arm))))
    return _result(env)


def handover_transfer(env, actor, *, from_arm, to_arm, middle_pose,
                      to_contact_point_id=None, to_pre_grasp_dis=0.07,
                      from_retreat_z=0.08):
    """Cross-arm transfer of an ALREADY-HELD actor (handover_block pattern): carry it to a
    middle pose without releasing, grasp with the receiving arm, release the source arm, retreat
    the source straight up. The receiving arm ends holding the actor."""
    bad = _entry(env)
    if bad:
        return bad
    env.move(env.place_actor(actor, target_pose=list(middle_pose), arm_tag=ArmTag(str(from_arm)),
                             functional_point_id=0, pre_dis=0.0, dis=0.0,
                             is_open=False, constrain="free"))
    if env.plan_success:
        kw = dict(arm_tag=ArmTag(str(to_arm)), pre_grasp_dis=float(to_pre_grasp_dis),
                  grasp_dis=0.0)
        if to_contact_point_id is not None:
            kw["contact_point_id"] = to_contact_point_id
        env.move(env.grasp_actor(actor, **kw))
    if env.plan_success:
        env.move(env.open_gripper(ArmTag(str(from_arm))))
    if env.plan_success and from_retreat_z:
        env.move(env.move_by_displacement(ArmTag(str(from_arm)), z=float(from_retreat_z)))
    return _result(env)


def drive_articulation_joint(env, actor, arm, *, target_qpos, joint_index=0,
                             grasp_contact_point_id, follow_contact_point_id,
                             pre_grasp_dis=0.08, max_iters=50, min_progress=1e-3):
    """Open/close an articulated joint by re-grasping a moving contact point until the joint
    reaches target_qpos (open_microwave door-follow pattern, WITHOUT its mid-episode
    check_success). Direction is inferred from the initial joint value; the loop stops on target
    reached, progress stall, plan failure, or max_iters."""
    bad = _entry(env)
    if bad:
        return bad

    def _q():
        return float(np.asarray(actor.get_qpos()).reshape(-1)[int(joint_index)])

    env.move(env.grasp_actor(actor, arm_tag=ArmTag(str(arm)),
                             pre_grasp_dis=float(pre_grasp_dis),
                             contact_point_id=grasp_contact_point_id))
    q0 = _q()
    direction = 1.0 if float(target_qpos) >= q0 else -1.0
    iters, stalled = 0, False
    while iters < int(max_iters) and env.plan_success:
        q = _q()
        if direction * (q - float(target_qpos)) >= 0.0:
            break
        env.move(env.grasp_actor(actor, arm_tag=ArmTag(str(arm)), pre_grasp_dis=0.0,
                                 grasp_dis=0.0, contact_point_id=follow_contact_point_id))
        iters += 1
        if direction * (_q() - q) <= float(min_progress):
            stalled = True
            break
    qf = _q()
    reached = direction * (qf - float(target_qpos)) >= 0.0
    out = _result(env, qpos_initial=round(q0, 4), qpos_final=round(qf, 4),
                  iters=iters, stalled=stalled, reached=reached)
    out["ok"] = out["ok"] and reached
    return out
