"""Single Python model/message/tool loop for the ``codeaction-reference`` scaffold."""
import base64
import json
import struct
import time
import zlib
from pathlib import Path

from codeaction.agents.reference.context import (
    IMAGE_POLICY_ID,
    IMAGE_POLICY_VERSION,
    ContextBudgetError,
    ContextManager,
    ContextPolicy,
)
from codeaction.runtime.episode import EpisodeRuntime, REFERENCE_EPISODE_CONTRACT
from codeaction.contracts.failures import (
    FailureCode,
    FailureOrigin,
    ProviderCallError,
    classify_provider_exception,
    default_failure,
)
from codeaction.contracts.identity import sha256_json
from codeaction.contracts.harness_parameters import declared_harness_parameters
from codeaction.interface.instructions import reference_instruction_surface
from codeaction.providers.model_adapter import (
    ModelAdapter,
    ScriptedModel,
    capabilities_of,
    invoke_provider,
    provider_request_profile,
    rate_limit_policy_of,
    request_profile_of,
    requested_output_tokens_for_profile,
    resolve_output_tokens,
    transport_profile_of,
)
from codeaction.providers.provider_runtime import normalize_rate_limit_policy, normalize_transport_profile
from codeaction.interface.schemas import PROGRAM_TOOL_NAMES, TOOL_SPECS, build_openai_tools, to_json
from codeaction.interface.api_reference import (
    API_REFERENCE_PATH,
    library_names,
    render_api_reference,
)
from codeaction.interface.tool_surface import (
    INTERFACE_REFERENCE,
    assert_surface_preflight,
    delivered_primitive_names,
    ordered_tool_names,
    resolve_interface_profile,
    surface_identity,
)
from codeaction.contracts.tool_results import (
    MODEL_VISIBLE_RESULT_MAX_BYTES,
    TOOL_RESULT_POLICY_ID,
    TOOL_RESULT_POLICY_VERSION,
)
from codeaction.contracts.version import TRANSCRIPT_SCHEMA_VERSION, git_version


# The per-turn output request every model starts from. Each provider adapter clamps it to that
# model's own declared native limit, so this value is deliberately at or above the largest limit any
# registered model declares: every model then requests exactly its native ceiling and the scaffold
# never decides who may write a longer turn. 8192 was binding rather than generous -- once Kimi's
# transport stopped killing long turns at 90 s it hit that ceiling exactly and lost whole episodes
# to a non-scoreable output_length_exceeded. `test_shared_output_request_never_binds` fails if a
# newly registered model declares more than this.
REQUESTED_OUTPUT_TOKENS = 131072
MAX_TOKENS_PER_TURN = REQUESTED_OUTPUT_TOKENS  # compatibility name; now the requested limit
DEFAULT_CONTEXT_POLICY = ContextPolicy()
FRAME_RETENTION = DEFAULT_CONTEXT_POLICY.reference_context_max_image_rounds
from codeaction.contracts.version import SCAFFOLD_NAME, SCAFFOLD_VERSION  # noqa: F401
MAX_IMAGE_BASE64_BYTES = 10 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def scaffold_card(capabilities=None, requested_output_tokens=REQUESTED_OUTPUT_TOKENS,
                  context_policy=DEFAULT_CONTEXT_POLICY, request_profile=None,
                  transport_profile=None, rate_limit_policy=None, implementation=None) -> dict:
    profile = request_profile or provider_request_profile("")
    capability_record = (
        capabilities.to_dict() if capabilities is not None
        else {"resolution": "per-model-before-first-turn"}
    )
    effective = (
        requested_output_tokens_for_profile(
            capabilities, requested_output_tokens, profile)
        if capabilities is not None else None
    )
    value = {
        "name": SCAFFOLD_NAME,
        "version": SCAFFOLD_VERSION,
        # Missing and null both mean the client sends no sampling override.  This records the
        # actual wire behavior rather than the old scaffold-level 0.0 constant.
        "temperature": profile.get("temperature"),
        "requested_output_tokens": int(effective or requested_output_tokens),
        "effective_output_tokens": effective,
        "max_tokens_per_turn": int(effective or requested_output_tokens),
        "frame_retention": context_policy.reference_context_max_image_rounds,
        "context_policy": context_policy.to_dict(),
        "tool_result_policy": {
            "id": TOOL_RESULT_POLICY_ID,
            "version": TOOL_RESULT_POLICY_VERSION,
            "max_bytes": MODEL_VISIBLE_RESULT_MAX_BYTES,
        },
        "image_policy": {
            "id": IMAGE_POLICY_ID,
            "version": IMAGE_POLICY_VERSION,
            "run_code_result_max_images":
                context_policy.run_code_result_max_images,
            "reference_context_max_image_rounds":
                context_policy.reference_context_max_image_rounds,
        },
        "provider_capabilities": capability_record,
        "provider_request_profile": profile,
        "provider_transport_profile": (
            transport_profile if transport_profile is not None else normalize_transport_profile()),
        "provider_rate_limit_policy": (
            rate_limit_policy if rate_limit_policy is not None
            else normalize_rate_limit_policy(None, quota_group=None)),
    }
    if implementation is not None:
        value["implementation"] = implementation
    value["config_sha256"] = sha256_json(value)
    return value


def _validate_png_bytes(raw: bytes) -> tuple[int, int]:
    """Validate the exact payload constraints Qwen inspects before inference."""
    if not isinstance(raw, bytes) or not raw.startswith(PNG_SIGNATURE):
        raise ValueError("reference image is not a PNG")
    cursor = len(PNG_SIGNATURE)
    chunks = []
    width = height = None
    saw_iend = False
    saw_chunk = False
    while cursor < len(raw):
        if cursor + 12 > len(raw):
            raise ValueError("reference PNG has a truncated chunk")
        length = struct.unpack(">I", raw[cursor:cursor + 4])[0]
        kind = raw[cursor + 4:cursor + 8]
        end = cursor + 12 + length
        if end > len(raw):
            raise ValueError("reference PNG chunk exceeds the payload")
        payload = raw[cursor + 8:cursor + 8 + length]
        expected_crc = struct.unpack(">I", raw[cursor + 8 + length:end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("reference PNG chunk CRC is invalid")
        if not saw_chunk and (kind != b"IHDR" or length != 13):
            raise ValueError("reference PNG does not start with a valid IHDR")
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT":
            chunks.append(payload)
        elif kind == b"IEND":
            saw_iend = True
            if end != len(raw):
                raise ValueError("reference PNG has bytes after IEND")
            break
        saw_chunk = True
        cursor = end
    if not saw_iend or width is None or height is None or not chunks:
        raise ValueError("reference PNG is missing IHDR, IDAT, or IEND")
    if width < 10 or height < 10 or max(width / height, height / width) > 200:
        raise ValueError("reference PNG dimensions are outside provider limits")
    try:
        zlib.decompress(b"".join(chunks))
    except zlib.error as exc:
        raise ValueError("reference PNG pixel stream cannot be decoded") from exc
    return width, height


def _image_block(image_ref):
    raw = Path(image_ref).read_bytes()
    _validate_png_bytes(raw)
    encoded = base64.b64encode(raw)
    if len(encoded) > MAX_IMAGE_BASE64_BYTES:
        raise ValueError("reference PNG exceeds the provider's Base64 payload limit")
    # validate=True proves the exact ASCII payload we are about to send is canonical Base64.
    base64.b64decode(encoded, validate=True)
    uri = "data:image/png;base64," + encoded.decode("ascii")
    return {"type": "image_url", "image_url": {"url": uri}}


def _projection_meta(call) -> dict:
    projection = call.projection
    return {
        "policy": projection.policy,
        "truncated": projection.truncated,
        "original_bytes": projection.original_bytes,
        "model_bytes": projection.model_bytes,
        "model_image_count": len(projection.model_image_refs),
        "all_image_count": len(projection.all_image_refs),
        "unique_model_image_count": len(set(projection.model_image_refs)),
        "duplicate_observation_ids_suppressed": (
            projection.duplicate_observation_ids_suppressed),
    }


def _output_limit_failure(requested, effective, declared_model_limit):
    reached_model_limit = int(effective) == int(declared_model_limit)
    return default_failure(
        FailureCode.OUTPUT_LENGTH_EXCEEDED,
        origin=(
            FailureOrigin.MODEL if reached_model_limit
            else FailureOrigin.HARNESS),
        scoreable=reached_model_limit,
        detail_safe=(
            f"requested_output_tokens={int(requested)}; "
            f"effective_output_tokens={int(effective)}; "
            f"declared_model_limit={int(declared_model_limit)}"),
    )


def run_episode(
    toolbox,
    task_text,
    model,
    out_dir,
    max_steps=40,
    wall_budget_s=900.0,
    physical_time_budget_s=900.0,
    sandbox=None,
    sim_step_fn=None,
    expected_tool_surface=None,
    contract_profile=REFERENCE_EPISODE_CONTRACT,
    requested_output_tokens=REQUESTED_OUTPUT_TOKENS,
    context_manager=None,
    expected_scaffold_config_sha256=None,
    runtime_override=None,
    tool_definitions=None,
    hybrid=None,
    run_code_max_internal_calls=None,
    transcript_sink=None,
    filesystem_policy=None,
    program_workspace_limits=None,
    harness_parameters=None,
):
    """Run one strictly sequential reference-scaffold attempt.

    ``runtime_override`` is the container seam: it may transport calls to the MCP episode server,
    but it must implement the same small runtime interface as ``EpisodeRuntime``.  The model,
    context, stop-reason, JSON-decoding, and sequential-call loop remain here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sandbox is not None:
        (out / "filesystem_audit.json").write_text(
            json.dumps(sandbox.filesystem_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    transcript_path = out / "transcript.jsonl"
    run_code_limit = int(
        run_code_max_internal_calls
        if run_code_max_internal_calls is not None
        else (getattr(sandbox, "max_tool_calls", 1) if sandbox is not None else 1)
    )
    profile_id = str((expected_tool_surface or {}).get("interface_profile")
                     or INTERFACE_REFERENCE)
    profile = resolve_interface_profile(profile_id)
    instruction_surface = reference_instruction_surface(
        task_text=task_text,
        max_tool_calls=int(max_steps),
        physical_time_budget_s=physical_time_budget_s,
        run_code_max_internal_calls=run_code_limit,
        composition_contract=(
            getattr(sandbox, "composition_contract", None)
            if sandbox is not None else None),
        harness_parameters=(
            harness_parameters if harness_parameters is not None
            else declared_harness_parameters(toolbox)),
        include_program_workspace=bool(profile.delivered_program_tools),
    )
    runtime = runtime_override or EpisodeRuntime(
        toolbox, task_text, profile=contract_profile, sandbox=sandbox,
        max_tool_calls=max_steps, wall_budget_s=wall_budget_s)
    registry = runtime.registry
    hybrid_enabled = (
        bool(hybrid) if hybrid is not None
        else (sandbox is not None or any(name in registry for name in PROGRAM_TOOL_NAMES))
    )
    if tool_definitions is None:
        delivered = set(delivered_primitive_names(profile_id))
        registered = [
            name for name in registry
            if name in TOOL_SPECS and name in delivered
            and name not in PROGRAM_TOOL_NAMES and name != "run_code"
        ]
        composition = (
            ["run_code", *profile.delivered_program_tools] if hybrid_enabled else [])
        tool_names = composition + registered + ["done"]
        tools = build_openai_tools(tool_names)
    else:
        tools = [dict(item) for item in tool_definitions]
        tool_names = [
            str((item.get("function") or {}).get("name") or "")
            for item in tools
        ]
        if any(not name for name in tool_names):
            raise ValueError("reference tool definitions must use OpenAI function-tool shape")
    # The allowlist is the surface we actually handed this provider, not the one the profile
    # would imply — those agree on every shipped profile, and where a caller supplies its own
    # definitions the delivered set is what it supplied. A caller-supplied runtime enforces its
    # own surface at its own boundary (the remote runtime's allowlist lives on the sim host).
    if runtime_override is None:
        runtime.restrict_to(tool_names)
    # Code-first hides the primitives from the schema surface, so their semantics must reach the
    # model some other way: the same TOOL_SPECS text, rendered into the agent's virtual workspace
    # before the first turn and readable with read_file / list_files.
    if sandbox is not None and profile.delivered_primitives is not None:
        sandbox.seed_file(
            API_REFERENCE_PATH,
            render_api_reference(library_names(
                delivered_primitive_names(profile_id),
                all_names=[name for name in registry if name in TOOL_SPECS],
            )),
        )
    observed_surface = None
    if expected_tool_surface is not None:
        observed_surface = surface_identity(
            profile_id,
            hybrid=hybrid_enabled,
            runtime_registry_names=registry,
        )
        assert_surface_preflight(expected_tool_surface, observed_surface)
        if tool_names != observed_surface["ordered_names"]:
            raise ValueError("reference tool definition order disagrees with expected surface")

    if not callable(getattr(model, "capabilities", None)):
        raise ValueError(
            "active codeaction-reference providers must declare ModelCapabilities")
    capabilities = capabilities_of(model)
    request_profile = request_profile_of(model)
    requested_output_tokens = requested_output_tokens_for_profile(
        capabilities, requested_output_tokens, request_profile)
    effective_output_tokens = resolve_output_tokens(capabilities, requested_output_tokens)
    context = context_manager or ContextManager(DEFAULT_CONTEXT_POLICY)
    scaffold = scaffold_card(
        capabilities, requested_output_tokens, context.policy,
        request_profile, transport_profile_of(model), rate_limit_policy_of(model))
    if expected_scaffold_config_sha256 is not None and \
            scaffold["config_sha256"] != str(expected_scaffold_config_sha256):
        raise ValueError(
            "reference scaffold config preflight mismatch: "
            f"expected={expected_scaffold_config_sha256}, "
            f"observed={scaffold['config_sha256']}")
    messages = []
    next_message_index = 1

    def append_message(role, *, anchor=False, **fields):
        nonlocal next_message_index
        message = {
            "role": role,
            "_message_id": f"m{next_message_index:06d}",
            **({"_context_anchor": True} if anchor else {}),
            **fields,
        }
        next_message_index += 1
        messages.append(message)
        return message

    append_message(
        "system", anchor=True, content=instruction_surface["system_prompt"])
    append_message(
        "user", anchor=True,
        content=[{"type": "text", "text": instruction_surface["task_message"]}])

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    # How much provider-outage time has already been handed back, so each turn credits only
    # its own delta rather than the running total.
    credited_infra_wait_s = 0.0
    no_tool_calls = 0
    turn_index = 0
    # How many turns actually came back with reasoning text.  Counted rather than inferred so a
    # run whose model was configured to think but never returned any of it is visible in the end
    # record, instead of only in whoever later greps the transcript for a null column.
    reasoning_turns = 0
    last_context_view = None
    text_compaction_count = 0
    text_compaction_first_turn = None
    text_compaction_dropped_messages = []
    text_compaction_dropped_seen = set()
    text_compaction_ledger_entries = 0
    visual_tail_trimmed_for_budget = False
    protected_units_eroded = 0
    # Compatibility counter: ANY reduction of the recorded conversation, image eviction included.
    context_reduced_any = False
    context_retry_count = 0
    image_transport_stats = {
        "image_blocks_sent": 0,
        "repeated_image_blocks_sent": 0,
        "duplicate_observation_ids_suppressed": 0,
        "seen_refs": set(),
    }

    def log(record):
        with transcript_path.open("a", encoding="utf-8") as stream:
            stream.write(to_json(record) + "\n")
        if transcript_sink is not None:
            transcript_sink(dict(record))

    def sim_step():
        if sim_step_fn is None:
            return None
        try:
            return int(sim_step_fn())
        except Exception:
            return None

    def record_context_view(view, turn):
        nonlocal text_compaction_count, text_compaction_first_turn
        nonlocal text_compaction_ledger_entries, visual_tail_trimmed_for_budget
        nonlocal protected_units_eroded, context_reduced_any
        context_reduced_any = context_reduced_any or view.compacted
        protected_units_eroded = max(protected_units_eroded, int(view.protected_units_eroded))
        if view.text_compacted:
            text_compaction_count += 1
            if text_compaction_first_turn is None:
                text_compaction_first_turn = int(turn)
            for message_id in view.dropped_message_ids:
                if message_id not in text_compaction_dropped_seen:
                    text_compaction_dropped_seen.add(message_id)
                    text_compaction_dropped_messages.append(message_id)
            text_compaction_ledger_entries = max(
                text_compaction_ledger_entries,
                int(view.text_compaction_ledger_entries),
            )
        visual_tail_trimmed_for_budget = (
            visual_tail_trimmed_for_budget or view.visual_tail_trimmed_for_budget)

    set_event_sink = getattr(model, "set_event_sink", None)
    if callable(set_event_sink):
        set_event_sink(lambda record: log({"turn": turn_index, **record}))

    def attach_images(projection):
        image_transport_stats["duplicate_observation_ids_suppressed"] += int(
            projection.duplicate_observation_ids_suppressed)
        for group in projection.image_groups:
            content = [{"type": "text", "text": group.text}]
            labels = group.labels or tuple(
                f"image {index + 1}/{len(group.refs)}"
                for index in range(len(group.refs)))
            for label, ref in zip(labels, group.refs):
                image_transport_stats["image_blocks_sent"] += 1
                if ref in image_transport_stats["seen_refs"]:
                    image_transport_stats["repeated_image_blocks_sent"] += 1
                else:
                    image_transport_stats["seen_refs"].add(ref)
                content.extend([
                    {"type": "text", "text": label},
                    _image_block(ref),
                ])
            append_message("user", content=content)

    log({
        "event": "meta",
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "scaffold": scaffold,
        "task": task_text,
        "tools": tool_names,
        "frame_retention": context.policy.reference_context_max_image_rounds,
        "temperature": scaffold.get("temperature"),
        "budget_unit": "tool_calls",
        "budget_visible": True,
        "run_code_internal_charged": False,
        "run_program_internal_charged": False,
        "filesystem_policy": (
            filesystem_policy
            if filesystem_policy is not None
            else ("structurally_denied" if sandbox is not None else None)),
        "program_workspace_limits": (
            program_workspace_limits
            if program_workspace_limits is not None
            else (getattr(sandbox, "program_limits", None)
                  if sandbox is not None else None)),
        "sim_step_clock": "physics_steps" if sim_step_fn is not None else None,
        "max_steps": int(max_steps),
        "max_tool_calls": int(max_steps),
        "physical_time_budget_s": physical_time_budget_s,
        "budget_contract_version": contract_profile.contract_version,
        "tool_surface": observed_surface,
        "instruction_surface": {
            key: instruction_surface[key] for key in (
                "instruction_contract_sha256", "instruction_surface_sha256",
                "fragment_ids", "fragment_manifest")
        },
        "run_code_max_calls": run_code_limit if hybrid_enabled else None,
        "model": getattr(model, "model", type(model).__name__),
        "model_capabilities": capabilities.to_dict(),
        "requested_output_tokens": int(requested_output_tokens),
        "effective_output_tokens": effective_output_tokens,
        "context_policy": context.policy.to_dict(),
        "tool_result_policy": {
            "id": TOOL_RESULT_POLICY_ID,
            "version": TOOL_RESULT_POLICY_VERSION,
            "max_bytes": MODEL_VISIBLE_RESULT_MAX_BYTES,
        },
        "image_policy": {
            "run_code_result_max_images":
                context.policy.run_code_result_max_images,
            "reference_context_max_image_rounds":
                context.policy.reference_context_max_image_rounds,
        },
        **git_version(__file__),
    })

    ended = False
    while not ended and not runtime.over:
        if not runtime.check_wall_budget():
            break
        turn_index += 1
        call_started = time.time()
        context_retry = False
        try:
            last_context_view = context.prepare(
                messages, capabilities, requested_output_tokens)
            record_context_view(last_context_view, turn_index)
            set_token_estimate = getattr(model, "set_request_token_estimate", None)
            if callable(set_token_estimate):
                set_token_estimate(
                    last_context_view.estimated_input_tokens, effective_output_tokens)
            try:
                model_turn = invoke_provider(
                    model, list(last_context_view.messages), tools,
                    requested_output_tokens)
            except ProviderCallError as exc:
                if exc.failure.code != FailureCode.CONTEXT_LENGTH_EXCEEDED:
                    raise
                context_retry = True
                last_context_view = context.prepare(
                    messages, capabilities, requested_output_tokens,
                    force_compact=True)
                record_context_view(last_context_view, turn_index)
                if last_context_view.force_compact_unreducible:
                    # The request is already at the floor the context policy will not go below, so
                    # a retry would resend the exact bytes the provider just rejected. No provider
                    # call is made and no retry is counted; the provider's own failure is reported.
                    log({
                        "event": "context_retry_declined",
                        "turn": turn_index,
                        "failure": exc.failure.to_dict(),
                        **last_context_view.event(),
                    })
                    raise
                context_retry_count += 1
                if callable(set_token_estimate):
                    set_token_estimate(
                        last_context_view.estimated_input_tokens, effective_output_tokens)
                log({
                    "event": "context_retry",
                    "turn": turn_index,
                    "failure": exc.failure.to_dict(),
                    **last_context_view.event(),
                })
                model_turn = invoke_provider(
                    model, list(last_context_view.messages), tools,
                    requested_output_tokens)
        except ContextBudgetError as exc:
            failure = default_failure(
                FailureCode.CONTEXT_LENGTH_EXCEEDED,
                detail_safe=f"ContextBudgetError: {exc}")
            runtime.done_report = failure.detail_safe
            runtime.finalize("context_length_exceeded", failure=failure)
            log({
                "event": "context_length_exceeded",
                "turn": turn_index,
                "failure": failure.to_dict(),
            })
            break
        except ProviderCallError as exc:
            runtime.done_report = str(exc)
            terminal_status = (
                "context_length_exceeded"
                if exc.failure.code == FailureCode.CONTEXT_LENGTH_EXCEEDED
                else "endpoint_failure")
            runtime.finalize(terminal_status, failure=exc.failure)
            log({
                "event": terminal_status,
                "turn": turn_index,
                "error": str(exc),
                "failure": exc.failure.to_dict(),
                "context_retry": context_retry,
            })
            break
        except RuntimeError as exc:
            failure = classify_provider_exception(exc)
            safe_error = f"provider call failed: {failure.code.value}"
            runtime.done_report = safe_error
            runtime.finalize("endpoint_failure", failure=failure)
            log({
                "event": "endpoint_failure",
                "turn": turn_index,
                "error": safe_error,
                "failure": failure.to_dict(),
                "context_retry": context_retry,
            })
            break

        usage = model_turn.usage
        for key, value in usage.items():
            if value is not None:
                usage_total[key] = usage_total.get(key, 0) + int(value)
        if str(model_turn.reasoning_content or "").strip():
            reasoning_turns += 1
        # Hand back provider retry/backoff/quota time before any stop-reason branch or no-tool-call
        # recovery. These waits are transcript telemetry only; they are never inserted into the
        # model-visible message list.
        infra_wait_total = float(getattr(model, "infra_wait_s", 0.0) or 0.0)
        if infra_wait_total > credited_infra_wait_s:
            credited = runtime.credit_wall_budget(infra_wait_total - credited_infra_wait_s)
            credited_infra_wait_s += credited
        latency = round(time.time() - call_started, 2)
        log({
            "event": "model_turn",
            "turn": turn_index,
            "stop_reason": model_turn.stop_reason,
            "stop_details": model_turn.stop_details,
            "message": model_turn.message,
            "reasoning_content": model_turn.reasoning_content,
            "usage": usage,
            "provider_request_id": model_turn.provider_request_id,
            "model_latency_s": latency,
            "context_retry": context_retry,
            "context": last_context_view.event(),
        })

        # Stop reason is checked before message insertion, argument decoding, or runtime dispatch.
        if model_turn.stop_reason == "length":
            failure = _output_limit_failure(
                requested_output_tokens,
                effective_output_tokens,
                capabilities.max_output_tokens,
            )
            runtime.done_report = failure.detail_safe
            runtime.finalize("output_length_exceeded", failure=failure)
            log({
                "event": "output_length_exceeded",
                "turn": turn_index,
                "requested_output_tokens": int(requested_output_tokens),
                "effective_output_tokens": effective_output_tokens,
                "failure": failure.to_dict(),
            })
            break
        # A policy refusal is the model declining the task, not the harness violating a contract.
        # It gets its own terminal status so the failure taxonomy attributes it to the model and a
        # report can separate "refused" from "produced an unusable turn".
        if model_turn.stop_reason == "refusal":
            details = model_turn.stop_details or {}
            failure = default_failure(
                FailureCode.MODEL_REFUSED,
                detail_safe=f"category={details.get('category') or 'unspecified'}")
            runtime.done_report = failure.detail_safe
            runtime.finalize("model_refusal", failure=failure)
            log({
                "event": "model_refusal",
                "turn": turn_index,
                "stop_details": model_turn.stop_details,
                "failure": failure.to_dict(),
            })
            break
        if model_turn.stop_reason not in ("tool_calls", "stop"):
            failure = default_failure(
                FailureCode.HARNESS_CONTRACT_VIOLATION,
                stage="model_turn",
                detail_safe=f"unsupported stop_reason={model_turn.stop_reason}")
            runtime.done_report = failure.detail_safe
            runtime.finalize("unsupported_stop_reason", failure=failure)
            log({
                "event": "unsupported_stop_reason",
                "turn": turn_index,
                "stop_reason": model_turn.stop_reason,
                "failure": failure.to_dict(),
            })
            break

        tool_calls = model_turn.message.get("tool_calls") or []
        if not tool_calls:
            no_tool_calls += 1
            content = model_turn.message.get("content") or ""
            log({
                "event": "no_tool_call",
                "turn": turn_index,
                "text": str(content)[:2000],
            })
            append_message("assistant", content=content)
            if no_tool_calls >= 3:
                runtime.finalize("no_tool_calls")
                break
            append_message("user", content="Respond with tool calls.")
            continue

        no_tool_calls = 0
        append_message(
            "assistant",
            content=model_turn.message.get("content"),
            tool_calls=tool_calls,
        )
        # Images are held until every tool result for THIS assistant turn has been appended. The
        # OpenAI-compatible protocol requires the `tool` messages answering one `tool_calls` array
        # to be contiguous, and attaching an image inside the loop inserted a `user` message
        # between two of them: Moonshot rejects that request outright (measured 2026-08-09, HTTP
        # 400 naming the unanswered tool_call_id) and every other vendor merely tolerated a history
        # that was already malformed.
        pending_images = []
        for call_index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "")
            arguments_text = str(function.get("arguments") or "{}")
            try:
                arguments = json.loads(arguments_text)
            except Exception:
                arguments = None


            sim_step_start = sim_step()
            call = runtime.dispatch(
                name, arguments, malformed_arguments=arguments is None)
            if call.kind == "done":
                log({
                    "event": "done",
                    "step": call.step,
                    "turn": turn_index,
                    "call_index": call_index,
                    "charged": call.charged,
                    "budget_used": runtime.budget_used,
                    "total_calls": runtime.total_calls,
                    "sim_step": sim_step(),
                    "report": runtime.done_report,
                    "unexecuted": [
                        str((item.get("function") or {}).get("name") or "")
                        for item in tool_calls[call_index + 1:]
                    ],
                })
                ended = True
                break
            append_message(
                "tool",
                tool_call_id=str(tool_call.get("id") or ""),
                content=to_json(call.payload),
            )
            if call.kind == "fatal":
                runtime.done_report = call.error_detail
                log({
                    "event": "episode_fatal",
                    "step": call.step,
                    "turn": turn_index,
                    "tool": name,
                    "error": call.error_detail.split(": ", 1)[-1],
                    "failure": call.failure.to_dict(),
                    "charged": call.charged,
                    "budget_used": runtime.budget_used,
                    "total_calls": runtime.total_calls,
                    "sim_step_start": sim_step_start,
                    "sim_step_end": sim_step(),
                })
                ended = True
                break
            if call.kind == "terminal":
                log({
                    "event": runtime.status,
                    "step": call.step,
                    "turn": turn_index,
                    "tool": name,
                    "args": arguments,
                    "result": call.projection.full_payload,
                    "model_result": call.payload,
                    "result_projection": _projection_meta(call),
                    "error": call.error_detail,
                    "failure": call.failure.to_dict(),
                    "charged": call.charged,
                    "budget_used": runtime.budget_used,
                    "total_calls": runtime.total_calls,
                    "sim_step_start": sim_step_start,
                    "sim_step_end": sim_step(),
                })
                ended = True
                break

            pending_images.append(call.projection)
            record = {
                "event": "limit" if call.kind == "limit" else "tool",
                "step": call.step,
                "turn": turn_index,
                "call_index": call_index,
                "tool": name,
                "args": arguments,
                "result": call.projection.full_payload,
                "model_result": call.payload,
                "result_projection": _projection_meta(call),
                "charged": call.charged,
                "budget_used": runtime.budget_used,
                "total_calls": runtime.total_calls,
                "sim_step_start": sim_step_start,
                "sim_step_end": sim_step(),
                "infra_retries": getattr(model, "infra_retries", 0),
            }
            if call.failure is not None:
                record["failure"] = call.failure.to_dict()
            if call_index == 0:
                record["usage"] = usage
                record["model_text"] = str(
                    model_turn.message.get("content") or "")[:4000]
                record["reasoning_content"] = model_turn.reasoning_content
                record["model_latency_s"] = latency
                record["stop_reason"] = model_turn.stop_reason
                record["provider_request_id"] = model_turn.provider_request_id
            log(record)
            if call.kind in ("limit", "control_wedge"):
                ended = True
                break
            if call.kind == "recoverable_abort":
                for cancelled_index, cancelled_tool_call in enumerate(
                        tool_calls[call_index + 1:], start=call_index + 1):
                    cancelled_function = cancelled_tool_call.get("function") or {}
                    cancelled_name = str(cancelled_function.get("name") or "")
                    cancelled_arguments_text = str(
                        cancelled_function.get("arguments") or "{}")
                    try:
                        cancelled_arguments = json.loads(cancelled_arguments_text)
                    except Exception:
                        cancelled_arguments = None
                    cancelled = runtime.cancel_after_recoverable_abort(
                        cancelled_name, cancelled_arguments)
                    append_message(
                        "tool",
                        tool_call_id=str(cancelled_tool_call.get("id") or ""),
                        content=to_json(cancelled.payload),
                    )
                    log({
                        "event": "tool_cancelled_after_action_abort",
                        "step": cancelled.step,
                        "turn": turn_index,
                        "call_index": cancelled_index,
                        "tool": cancelled_name,
                        "args": cancelled_arguments,
                        "result": cancelled.projection.full_payload,
                        "charged": cancelled.charged,
                        "budget_used": runtime.budget_used,
                        "total_calls": runtime.total_calls,
                        "sim_step_start": sim_step(),
                        "sim_step_end": sim_step(),
                    })
                break

        for projection in pending_images:
            attach_images(projection)

    if not runtime.over:
        runtime.finalize("budget_exhausted")
    image_messages_in_context = 0
    image_blocks_in_context = 0
    context_image_urls = set()
    if last_context_view is not None:
        for message in last_context_view.messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            blocks = [
                item for item in content
                if isinstance(item, dict) and item.get("type") == "image_url"
            ]
            if blocks:
                image_messages_in_context += 1
            image_blocks_in_context += len(blocks)
            for block in blocks:
                url = (block.get("image_url") or {}).get("url")
                if isinstance(url, str):
                    context_image_urls.add(url)
    stats = {
        "status": runtime.status,
        "done_report": runtime.done_report,
        "steps": runtime.budget_used,
        "budget_used": runtime.budget_used,
        "total_calls": runtime.total_calls,
        "turns": turn_index,
        "tool_calls_used": runtime.budget_used,
        "tool_call_budget": int(max_steps),
        "total_tool_dispatches": runtime.total_calls,
        "model_turns": turn_index,
        "reasoning_turns": reasoning_turns,
        "infra_retries": getattr(model, "infra_retries", 0),
        "infra_wait_s": round(float(getattr(model, "infra_wait_s", 0.0) or 0.0), 3),
        "quota_wait_s": round(float(getattr(model, "quota_wait_s", 0.0) or 0.0), 3),
        "wall_credit_s": round(float(getattr(runtime, "wall_credit_s", 0.0) or 0.0), 3),
        "usage": usage_total,
        "wall_s": runtime.wall_s,
        # Compatibility field: historically this counted image-bearing messages, not image
        # blocks.  Keep it stable and expose the unambiguous counters alongside it.
        "images_in_context": image_messages_in_context,
        "image_messages_in_context": image_messages_in_context,
        "image_blocks_in_context": image_blocks_in_context,
        "unique_images_in_context": len(context_image_urls),
        "image_blocks_sent": image_transport_stats["image_blocks_sent"],
        "unique_images_sent": len(image_transport_stats["seen_refs"]),
        "repeated_image_blocks_sent": image_transport_stats["repeated_image_blocks_sent"],
        "duplicate_observation_ids_suppressed": (
            image_transport_stats["duplicate_observation_ids_suppressed"]),
        "text_history_estimated_tokens": (
            last_context_view.text_history_estimated_tokens
            if last_context_view is not None else 0),
        "protected_context_estimated_tokens": (
            last_context_view.protected_context_estimated_tokens
            if last_context_view is not None else 0),
        "recent_exact_context_estimated_tokens": (
            last_context_view.recent_exact_context_estimated_tokens
            if last_context_view is not None else 0),
        "visual_tail_estimated_tokens": (
            last_context_view.visual_tail_estimated_tokens
            if last_context_view is not None else 0),
        "visual_tail_image_count": (
            last_context_view.visual_tail_image_count
            if last_context_view is not None else 0),
        "visual_tail_source_message_ids": (
            list(last_context_view.visual_tail_source_message_ids)
            if last_context_view is not None else []),
        "visual_tail_duplicate_images_suppressed": (
            last_context_view.visual_tail_duplicate_images_suppressed
            if last_context_view is not None else 0),
        "visual_tail_trimmed_for_budget": visual_tail_trimmed_for_budget,
        "transcript": str(transcript_path),
        "failure": runtime.failure.to_dict() if runtime.failure is not None else None,
        # Non-null when the hidden finalize control never reached the simulator. The attempt then
        # ends on two disagreeing sides, and this is the only thing that says the cause was
        # transport rather than the episode. Absent on runtimes without the control.
        "finalize_control_error": getattr(runtime, "finalize_control_error", None),
        "requested_output_tokens": int(requested_output_tokens),
        "effective_output_tokens": effective_output_tokens,
        "text_compacted": text_compaction_count > 0,
        "text_compaction_count": text_compaction_count,
        "text_compaction_first_turn": text_compaction_first_turn,
        "text_compaction_dropped_messages": text_compaction_dropped_messages,
        "text_compaction_ledger_entries": text_compaction_ledger_entries,
        "protected_units_eroded": protected_units_eroded,
        "context_retry_count": context_retry_count,
        # Compatibility field: keeps its historical meaning -- ANY reduction of the recorded
        # conversation, image eviction included. `text_compacted` is the narrower new fact.
        "context_compacted": context_reduced_any,
        "scaffold": scaffold,
        "model_capabilities": capabilities.to_dict(),
        "context_policy": context.policy.to_dict(),
        "tool_result_policy": scaffold["tool_result_policy"],
        "image_policy": scaffold["image_policy"],
    }
    log({"event": "end", **stats})
    try:
        # The turn-by-turn human-readable projection ships with every episode, no extra step.
        # It is derived from the transcript just written, so its failure must never fail the
        # episode; the error lands in a sidecar file instead of the transcript (whose final
        # event stays "end").
        from codeaction.reporting.turns_view import write_turns_view
        write_turns_view(out)
    except Exception as exc:
        (out / "turns.v1.error.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n", encoding="utf-8")
    return stats


__all__ = [
    "DEFAULT_CONTEXT_POLICY",
    "FRAME_RETENTION",
    "MAX_TOKENS_PER_TURN",
    "ModelAdapter",
    "REQUESTED_OUTPUT_TOKENS",
    "SCAFFOLD_NAME",
    "SCAFFOLD_VERSION",
    "ScriptedModel",
    "run_episode",
    "scaffold_card",
]
