"""The batch write guard: a write-class call issued in the same parallel tool
batch as a read-class call is refused at the transport, with the reason steered
into the turn first.

A model that sends a read and a write that quotes the read's answer in ONE
batch has not seen the answer yet: the calls run in parallel, so the write
carries a guessed value. The guard refuses the write and tells the model to
send it again once the read has returned. It is opt-in
(``hooks.batch_write_guard``), so with the setting off nothing changes.

Three layers, each pinned here:

* the classifier (:func:`batch_guard.classify`) and its explicit tables;
* the batch window (:class:`batch_guard.BatchWindow`): which calls count as one
  batch;
* both transports' ``approve_tool``, fed by real ``tool_call`` frames, steering
  the notice before the rejection goes on the wire.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp.types import (
    EVENT_PERMISSION_REQUEST,
    EVENT_TOOL_CALL,
    OPTION_ALLOW_ONCE,
    OUTCOME_CANCELLED,
    OUTCOME_SELECTED,
    AcpEvent,
    JsonRpcMessage,
)


def _guard():
    # Resolved by name so the unfixed tree fails inside each test, naming the
    # missing module, rather than at collection.
    try:
        return importlib.import_module("kiro_crew.acp.batch_guard")
    except ImportError:
        pytest.fail("kiro_crew.acp.batch_guard is missing: no batch write guard exists")


# ── frames ──


def _tool_call_frame(
    call_id: str,
    tool_name: str,
    raw_input: dict,
    *,
    kind: str = "",
    server: str = "",
    title: str = "",
) -> JsonRpcMessage:
    kiro: dict = {"toolName": tool_name}
    if server:
        kiro["mcpServerName"] = server
    update: dict = {
        "sessionUpdate": "tool_call",
        "toolCallId": call_id,
        "title": title or tool_name,
        "rawInput": raw_input,
        "_meta": {"kiro": kiro},
    }
    if kind:
        update["kind"] = kind
    return JsonRpcMessage(method="session/update", params={"sessionId": "s1", "update": update})


def _settled_frame(call_id: str, status: str = "completed") -> JsonRpcMessage:
    return JsonRpcMessage(
        method="session/update",
        params={
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": status,
                "content": [{"type": "content", "content": {"type": "text", "text": "ok"}}],
            },
        },
    )


def _read_frame(call_id: str = "tc-read") -> JsonRpcMessage:
    return _tool_call_frame(
        call_id,
        "shell",
        {"command": "gh pr view 7 --json headRefOid"},
        kind="execute",
        title="Running: gh pr view 7",
    )


def _write_frame(call_id: str = "tc-write") -> JsonRpcMessage:
    return _tool_call_frame(
        call_id,
        "shell",
        {"command": "gh pr merge 7 --squash"},
        kind="execute",
        title="Running: gh pr merge 7 --squash",
    )


def _write_permission(request_id: int, call_id: str = "tc-write") -> AcpEvent:
    command = "gh pr merge 7 --squash"
    return AcpEvent(
        kind=EVENT_PERMISSION_REQUEST,
        request_id=request_id,
        tool_call_id=call_id,
        title=f"Running: {command}",
        tool_kind="execute",
        is_shell=True,
        tool_input=json.dumps({"command": command}),
    )


def _event(**fields) -> AcpEvent:
    return AcpEvent(kind=EVENT_TOOL_CALL, **fields)


def _shell(command: str) -> AcpEvent:
    return _event(is_shell=True, tool_name="shell", tool_input=json.dumps({"command": command}))


# ── classifier ──


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 12 --squash --admin --match-head-commit abc",
        "gh pr create --title t --body-file ./b.md",
        "gh pr review 9 --approve -F body.md",
        "gh pr comment 9 --body hi",
        "gh issue comment 3 --body hi",
        "git push -u origin feat/x",
        "git -C /w commit -m 'msg'",
        "cd /w && git commit -F msg.txt",
        "cr --new-review",
        "aws --profile p --region us-west-2 dynamodb put-item --table-name t",
        "aws s3 cp a.txt s3://bucket/a.txt",
        "aws lambda invoke --function-name f out.json",
        "gh api -X POST repos/o/r/issues",
    ],
)
def test_clearly_write_shell_commands_are_write_class(command):
    guard = _guard()
    assert guard.classify(_shell(command)) == guard.WRITE


@pytest.mark.parametrize(
    "command",
    [
        "gh pr view 12 --json state,mergeCommit",
        "gh pr checks 12",
        "git log -1 --oneline",
        "git status -sb",
        "aws sts get-caller-identity",
        "aws --profile p ec2 describe-instances",
        "aws s3 ls s3://bucket/",
        "aws logs filter-log-events --log-group-name g",
        "ls -la && cat README.md",
        "python3 -m pytest -q test/test_x.py",
        "echo cr is a word here",
    ],
)
def test_every_other_shell_command_is_read_class(command):
    guard = _guard()
    assert guard.classify(_shell(command)) == guard.READ


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("fs_write", "write"),
        ("write", "write"),
        ("str_replace", "write"),
        ("fs_append", "write"),
        ("delete_file", "write"),
        ("fs_read", "read"),
        ("read", "read"),
        ("read_file", "read"),
        ("list_directory", "read"),
        ("grep", "read"),
        ("grep_search", "read"),
        ("glob", "read"),
        ("file_search", "read"),
        ("web_fetch", "read"),
    ],
)
def test_built_in_file_tools_follow_the_table(tool_name, expected):
    guard = _guard()
    assert guard.classify(_event(tool_name=tool_name)) == expected


@pytest.mark.parametrize(
    ("server", "tool", "expected"),
    [
        ("kirocrew-work", "work_report", "write"),
        ("kirocrew-work", "work_ledger_record", "write"),
        ("kirocrew-work", "work_ledger_read", "read"),
        ("kirocrew-work", "work_brief", "read"),
        ("kirocrew-dashboard", "session_send", "write"),
        ("kirocrew-dashboard", "session_broadcast", "write"),
        ("kirocrew-dashboard", "session_status", "read"),
        ("kirocrew-dashboard", "session_read_message", "read"),
        ("kirocrew-core", "send_message", "write"),
        ("kirocrew-core", "session_ledger_record", "write"),
        ("kirocrew-core", "memory_recall", "read"),
        ("kirocrew-core", "spawn_status", "read"),
    ],
)
def test_first_party_mcp_tools_follow_the_table(server, tool, expected):
    guard = _guard()
    event = _event(tool_name=tool, mcp_server_name=server, mcp_identity_trusted=True)
    assert guard.classify(event) == expected


def test_an_unknown_tool_is_not_guarded():
    # Unknown means "neither": the guard never refuses more than its tables name.
    guard = _guard()
    assert guard.classify(_event(tool_name="frobnicate", mcp_server_name="third-party")) == ""
    assert guard.classify(_event(title="something")) == ""


# ── batch window ──


def test_a_write_in_a_batch_with_a_read_is_refused():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    window.dispatched("w", guard.WRITE)
    reason = window.refusal_for("w", guard.WRITE)
    assert reason is not None
    assert "same parallel tool batch" in reason
    assert "next step" in reason


def test_the_read_itself_is_never_refused():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    window.dispatched("w", guard.WRITE)
    assert window.refusal_for("r", guard.READ) is None


def test_a_read_that_returned_before_the_write_was_sent_is_a_different_batch():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    window.settled("r")
    window.dispatched("w", guard.WRITE)
    assert window.refusal_for("w", guard.WRITE) is None


def test_a_read_that_settles_while_the_write_waits_still_shares_its_batch():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    window.dispatched("w", guard.WRITE)
    window.settled("r")
    assert window.refusal_for("w", guard.WRITE) is not None


def test_writes_alone_and_unknown_peers_pass():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("w1", guard.WRITE)
    window.dispatched("w2", guard.WRITE)
    window.dispatched("x", "")
    assert window.refusal_for("w2", guard.WRITE) is None


def test_a_call_the_window_never_saw_is_not_judged():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    assert window.refusal_for("never-announced", guard.WRITE) is None


def test_a_new_turn_starts_an_empty_window():
    guard = _guard()
    window = guard.BatchWindow()
    window.dispatched("r", guard.READ)
    window.reset()
    window.dispatched("w", guard.WRITE)
    assert window.refusal_for("w", guard.WRITE) is None


def test_the_window_is_bounded():
    guard = _guard()
    window = guard.BatchWindow()
    for i in range(guard.MAX_BATCH_CALLS + 50):
        window.dispatched(f"r{i}", guard.READ)
    assert len(window) <= guard.MAX_BATCH_CALLS


# ── the setting ──


def test_the_guard_is_off_unless_the_operator_turns_it_on():
    from kiro_crew.hook_runtime.hook_types import HooksConfig

    assert HooksConfig.from_dict({}).batch_write_guard is False
    assert HooksConfig.from_dict({"batch_write_guard": True}).batch_write_guard is True
    # A malformed value keeps it off: the guard refuses calls, so junk must not
    # switch it on.
    assert HooksConfig.from_dict({"batch_write_guard": "yes"}).batch_write_guard is False
    assert HooksConfig(batch_write_guard=True).to_dict()["batch_write_guard"] is True


# ── transports ──


def _answers(send_mock):
    return [call.args[1]["outcome"] for call in send_mock.await_args_list]


def _client(tmp_path):
    from kiro_crew.acp.client import AcpClient

    client = AcpClient(work_dir=tmp_path)
    client._send_response = AsyncMock()
    client._send_error = AsyncMock()
    return client


def _handle():
    from kiro_crew.acp.session_handle import AcpSessionHandle

    runtime = MagicMock()
    runtime.send_response = AsyncMock()
    runtime.send_error = AsyncMock()
    return AcpSessionHandle("s1", asyncio.Queue(), runtime)


def _feed(transport, msg):
    if hasattr(transport, "_handle_update"):
        transport._handle_update(msg)
        return
    transport._extract_tool_event(msg)
    transport._extract_tool_call_update(msg)


def _with_steer(transport, monkeypatch):
    order: list[str] = []
    steered: list[str] = []

    async def _steer(text):
        order.append("steer")
        steered.append(text)
        return True

    send = (
        transport._send_response
        if hasattr(transport, "_send_response")
        else (transport._runtime.send_response)
    )

    async def _send(*args, **kwargs):
        order.append("answer")

    send.side_effect = _send
    monkeypatch.setattr(transport, "steer", _steer, raising=False)
    monkeypatch.setattr(type(transport), "supports_refusal_steer", property(lambda self: True))
    return order, steered


def _sent(transport):
    if hasattr(transport, "_send_response") and isinstance(transport._send_response, AsyncMock):
        return transport._send_response
    return transport._runtime.send_response


_TRANSPORTS = [
    pytest.param(lambda tmp_path: _client(tmp_path), id="AcpClient"),
    pytest.param(lambda tmp_path: _handle(), id="AcpSessionHandle"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_transport", _TRANSPORTS)
async def test_transport_refuses_a_write_batched_with_a_read(tmp_path, monkeypatch, make_transport):
    guard = _guard()
    monkeypatch.setattr(guard, "guard_enabled", lambda: True)
    transport = make_transport(tmp_path)
    order, steered = _with_steer(transport, monkeypatch)
    _feed(transport, _read_frame())
    _feed(transport, _write_frame())
    transport._permission_gate_events[41] = _write_permission(41)

    assert await transport.approve_tool(41) is False

    outcomes = _answers(_sent(transport))
    assert outcomes and all(o.get("optionId") != OPTION_ALLOW_ONCE for o in outcomes)
    assert outcomes[-1] == {"outcome": OUTCOME_CANCELLED}
    # The reason reaches the model in-band, BEFORE the rejection is written.
    assert order[:2] == ["steer", "answer"]
    assert "batch guard" in steered[0]
    assert "next step" in steered[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_transport", _TRANSPORTS)
async def test_transport_approves_the_same_write_once_the_read_returned(
    tmp_path, monkeypatch, make_transport
):
    guard = _guard()
    monkeypatch.setattr(guard, "guard_enabled", lambda: True)
    transport = make_transport(tmp_path)
    _feed(transport, _read_frame())
    _feed(transport, _settled_frame("tc-read"))
    _feed(transport, _write_frame())
    transport._permission_gate_events[42] = _write_permission(42)

    assert await transport.approve_tool(42) is True
    assert _answers(_sent(transport))[-1].get("outcome") == OUTCOME_SELECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("make_transport", _TRANSPORTS)
async def test_transport_with_the_guard_off_approves_as_before(
    tmp_path, monkeypatch, make_transport
):
    guard = _guard()
    monkeypatch.setattr(guard, "guard_enabled", lambda: False)
    transport = make_transport(tmp_path)
    _feed(transport, _read_frame())
    _feed(transport, _write_frame())
    transport._permission_gate_events[43] = _write_permission(43)

    assert await transport.approve_tool(43) is True


@pytest.mark.asyncio
async def test_guard_enabled_reads_the_hooks_setting(monkeypatch):
    guard = _guard()
    from kiro_crew.config import KiroCrewConfig

    class _Cfg:
        hooks = {"batch_write_guard": True}

    monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: _Cfg()))
    assert guard.guard_enabled() is True
    _Cfg.hooks = {}
    assert guard.guard_enabled() is False


def test_demo_refusal_text():
    """The refusal a model reads, printed for the PR body's demo transcript."""
    guard = _guard()
    from kiro_crew.deny_notice import build_refusal_steer_notice

    window = guard.BatchWindow()
    window.dispatched("tc-read", guard.READ)
    window.dispatched("tc-write", guard.WRITE)
    reason = window.refusal_for("tc-write", guard.WRITE)
    assert reason is not None
    notice = build_refusal_steer_notice(
        "Running: gh pr merge 7 --squash", reason, cause=guard.DENY_CAUSE_BATCH_READ_WRITE
    )
    print(notice)
    assert "batch guard" in notice
    assert "NOT a user action" in notice
