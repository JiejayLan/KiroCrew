"""The batch write guard: refuse a write-class tool call issued in the same
parallel tool batch as a read-class call.

A model that sends a read and a write in ONE batch has not seen the read's
answer when it writes the write's arguments: the calls run in parallel, so a sha,
an id or a status the write quotes is a guess. This guard refuses the write and
tells the model to send it again once the read has returned. It judges the SET
of calls, not one command, which is why it is not a ``GATE_TIERS`` row or a
denied-command rule: the gate judges one call with no session state.

Three parts:

* :func:`classify` -- read, write, or neither, from explicit tables. A tool the
  tables do not name is neither, so the guard never refuses more than it names
  and an unknown peer never causes a refusal.
* :class:`BatchWindow` -- which calls form one batch, fed by the transport from
  the ``tool_call`` frames and their terminal ``tool_call_update``. A call that
  arrives while no call is open starts a new batch; one that arrives while any
  call is open joins it; a member stays a member after it settles, until the
  next batch starts. A write the window never saw is not judged.
* :func:`screen_approval` -- called by both transports' ``approve_tool`` after
  the security floor. When it refuses, it has already steered the reason into
  the running turn (the request is still unanswered, so the steer is queued and
  not dropped); the transport then rejects through its floor-refusal branch.

Opt-in: ``hooks.batch_write_guard`` in config.json, default off (see
:func:`guard_settings`). ``hooks.batch_write_guard_commands`` (empty by default)
lists extra write-command prefixes for tools only a local install knows. It is a
discipline guard, not a security control: the security floor runs first and is
unchanged.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import Any

from kiro_crew.constants import DENY_CAUSE_BATCH_READ_WRITE
from kiro_crew.platform.tool_names import KIRO_CLI_NAME_BY_KAS_TOOL, KIRO_TOOL_ALIASES

logger = logging.getLogger(__name__)

__all__ = [
    "DENY_CAUSE_BATCH_READ_WRITE",
    "MAX_BATCH_CALLS",
    "READ",
    "WRITE",
    "BatchWindow",
    "classify",
    "classify_command",
    "guard_settings",
    "screen_approval",
]

READ = "read"
WRITE = "write"

#: Members one window keeps. A batch past this size stops admitting members, so
#: a backend that never sends terminal updates cannot grow the window unbounded.
MAX_BATCH_CALLS = 256

# ── Tables ──

#: Built-ins by the kiro-cli name (KAS ids and kiro-cli's short stamps are mapped
#: onto these first, through ``platform.tool_names``).
_WRITE_BUILTINS: frozenset[str] = frozenset({"fs_write", "fs_append", "str_replace", "delete_file"})
_READ_BUILTINS: frozenset[str] = frozenset(
    {
        "fs_read",
        "list_directory",
        "read_file",
        "grep",
        "grep_search",
        "glob",
        "file_search",
        "web_fetch",
        "web_search",
        "remote_web_search",
    }
)
_SHELL_BUILTINS: frozenset[str] = frozenset({"execute_bash", "execute_pwsh", "run_command"})

#: Kiro Crew's own MCP tools, keyed by ``(server, tool)``.
_WRITE_MCP_TOOLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("kirocrew-work", "work_report"),
        ("kirocrew-work", "work_ledger_record"),
        ("kirocrew-work", "work_ledger_rebuild"),
        ("kirocrew-dashboard", "session_send"),
        ("kirocrew-dashboard", "session_broadcast"),
        ("kirocrew-core", "send_message"),
        ("kirocrew-core", "session_ledger_record"),
    }
)
_READ_MCP_TOOLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("kirocrew-work", "work_brief"),
        ("kirocrew-work", "work_ledger_read"),
        ("kirocrew-dashboard", "session_status"),
        ("kirocrew-dashboard", "session_read_message"),
        ("kirocrew-dashboard", "chat_folder_tree"),
        ("kirocrew-core", "memory_recall"),
        ("kirocrew-core", "search_chat_history"),
        ("kirocrew-core", "get_chat_session"),
        ("kirocrew-core", "skill_search"),
        ("kirocrew-core", "list_sessions"),
        ("kirocrew-core", "session_ledger_read"),
        ("kirocrew-core", "spawn_status"),
        ("kirocrew-core", "spawn_list"),
        ("kirocrew-core", "resource_status"),
        ("kirocrew-core", "artifact_get"),
        ("kirocrew-core", "artifact_list"),
        ("kirocrew-core", "local_knowledge_search"),
        ("kirocrew-core", "workflow_status"),
        ("kirocrew-core", "workflow_result"),
        ("kirocrew-core", "monitor_inspect"),
    }
)

#: ACP ``kind`` fallback, read only when the frame carried no tool identity.
_READ_KINDS: frozenset[str] = frozenset({"read", "fetch", "search"})
_WRITE_KINDS: frozenset[str] = frozenset({"edit", "delete", "move"})

#: Shell commands that are clearly writes. Everything else a shell runs counts as
#: a read: it produces output a later call may quote.
_WRITE_SHELL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgh\s+(?:pr|issue)\s+(?:merge|create|review|comment|close|edit|ready|reopen)\b"),
    re.compile(r"\bgh\s+release\s+(?:create|edit|delete|upload)\b"),
    re.compile(r"\bgh\s+api\b.*?(?:-X|--method)[\s=]*(?:POST|PUT|PATCH|DELETE)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(?:push|commit)\b"),
)

#: ``aws <service> <op>`` operations that only read.
_AWS_READ_OP_PREFIXES: tuple[str, ...] = (
    "describe",
    "get",
    "list",
    "head",
    "query",
    "scan",
    "search",
    "lookup",
    "filter",
    "batch-get",
    "wait",
    "ls",
    "help",
    "validate",
)
#: aws global options that take a value, skipped when finding service and op.
_AWS_VALUED_OPTIONS: frozenset[str] = frozenset(
    {
        "--profile",
        "--region",
        "--output",
        "--endpoint-url",
        "--query",
        "--color",
        "--ca-bundle",
        "--cli-read-timeout",
        "--cli-connect-timeout",
    }
)
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")


def _segment_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _aws_is_write(words: list[str]) -> bool:
    positional: list[str] = []
    skip = False
    for word in words[1:]:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            skip = word in _AWS_VALUED_OPTIONS
            continue
        positional.append(word)
        if len(positional) == 2:
            break
    if len(positional) < 2:
        return False
    return not positional[1].lower().startswith(_AWS_READ_OP_PREFIXES)


def _prefix_words(extra_write_commands: tuple[str, ...]) -> list[list[str]]:
    prefixes = []
    for prefix in extra_write_commands:
        words = _segment_words(prefix.strip()) if isinstance(prefix, str) else []
        if words:
            prefixes.append(words)
    return prefixes


def classify_command(command: str, extra_write_commands: tuple[str, ...] = ()) -> str:
    """READ or WRITE for a shell command line.

    *extra_write_commands* are the operator's own write-command prefixes
    (``hooks.batch_write_guard_commands``), matched word by word against the
    start of each segment's program words, for tools this table does not know.
    """
    for pattern in _WRITE_SHELL_PATTERNS:
        if pattern.search(command):
            return WRITE
    prefixes = _prefix_words(extra_write_commands)
    for segment in _SEGMENT_SPLIT.split(command):
        words = _segment_words(segment.strip())
        # Skip leading env assignments and simple wrappers, so ``X=1 tool`` and
        # ``timeout 60 aws ...`` are judged by their program.
        while words and ("=" in words[0] and not words[0].startswith("-")):
            words = words[1:]
        while words and words[0] in ("timeout", "env", "nohup", "time"):
            words = words[1:]
            while words and (words[0].startswith("-") or words[0].replace(".", "").isdigit()):
                words = words[1:]
        if not words:
            continue
        program = words[0].rsplit("/", 1)[-1]
        program_words = [program, *words[1:]]
        if any(program_words[: len(p)] == p for p in prefixes):
            return WRITE
        if program == "aws" and _aws_is_write(words):
            return WRITE
    return READ


def _tool_identity(event: Any) -> tuple[str, str]:
    """``(server, tool)`` from the harness-stamped identity, never the title."""
    server = str(getattr(event, "mcp_server_name", "") or "").lstrip("@")
    name = str(getattr(event, "harness_builtin_tool", "") or getattr(event, "tool_name", "") or "")
    name = name.lstrip("@")
    if "/" in name:
        prefix, name = name.rsplit("/", 1)
        server = server or prefix
    return server, name


def _command_of(event: Any) -> str:
    try:
        command = event.shell_command
    except Exception:
        command = None
    return command if isinstance(command, str) else ""


def classify(event: Any, extra_write_commands: tuple[str, ...] = ()) -> str:
    """READ, WRITE, or ``""`` (not guarded) for one tool-call event.

    *extra_write_commands* reaches :func:`classify_command` for a shell call.
    """
    server, name = _tool_identity(event)
    if server:
        if (server, name) in _WRITE_MCP_TOOLS:
            return WRITE
        if (server, name) in _READ_MCP_TOOLS:
            return READ
        return ""
    if name:
        builtin = KIRO_TOOL_ALIASES.get(name, name)
        if builtin in _SHELL_BUILTINS:
            return classify_command(_command_of(event), extra_write_commands)
        if builtin in _WRITE_BUILTINS:
            return WRITE
        if builtin in _READ_BUILTINS:
            return READ
        policy_name = KIRO_CLI_NAME_BY_KAS_TOOL.get(builtin, "")
        if policy_name in _READ_BUILTINS:
            return READ
        if policy_name in _WRITE_BUILTINS:
            return WRITE
        return ""
    if getattr(event, "is_shell", False):
        return classify_command(_command_of(event), extra_write_commands)
    kind = str(getattr(event, "tool_kind", "") or "")
    if kind in _READ_KINDS:
        return READ
    if kind in _WRITE_KINDS:
        return WRITE
    return ""


# ── Window ──


class BatchWindow:
    """The calls of the current parallel batch on one session, by tool-call id."""

    def __init__(self) -> None:
        self._members: dict[str, str] = {}
        self._open: set[str] = set()

    def __len__(self) -> int:
        return len(self._members)

    def dispatched(self, call_id: str, klass: str) -> None:
        """A ``tool_call`` frame announced *call_id* with class *klass*."""
        if not call_id:
            return
        if call_id in self._members:
            self._open.add(call_id)
            return
        if not self._open:
            self._members.clear()
        if len(self._members) >= MAX_BATCH_CALLS:
            return
        self._members[call_id] = klass
        self._open.add(call_id)

    def settled(self, call_id: str) -> None:
        """*call_id* reached a terminal status. It stays a member of its batch."""
        self._open.discard(call_id)

    def reset(self) -> None:
        """A new prompt turn: nothing carries over."""
        self._members.clear()
        self._open.clear()

    def read_peers(self, call_id: str) -> int:
        """How many OTHER members of *call_id*'s batch are reads (0 when unseen)."""
        if call_id not in self._members:
            return 0
        return sum(1 for cid, k in self._members.items() if cid != call_id and k == READ)

    def refusal_for(self, call_id: str, klass: str) -> str | None:
        """The refusal for approving *call_id* (class *klass*), or ``None``."""
        if klass != WRITE:
            return None
        reads = self.read_peers(call_id)
        if not reads:
            return None
        return (
            "Blocked by the batch guard: this write-class call was issued in the same "
            f"parallel tool batch as {reads} read-class call(s), so it may carry a value "
            "those reads had not returned yet. Nothing ran. Wait for the reads to return, "
            "check their output, then send this call again on its own in your next step."
        )


def note_dispatched(window: BatchWindow | None, event: Any) -> None:
    """Feed a ``tool_call`` event into *window*. Never raises."""
    if window is None:
        return
    try:
        window.dispatched(str(getattr(event, "tool_call_id", "") or ""), classify(event))
    except Exception:
        logger.debug("batch guard: could not classify a tool call", exc_info=True)


def note_settled(window: BatchWindow | None, call_id: str) -> None:
    if window is not None and call_id:
        window.settled(call_id)


# ── Approval ──


def guard_settings() -> tuple[bool, tuple[str, ...]]:
    """``(enabled, extra write-command prefixes)`` from the ``hooks`` section.

    ``hooks.batch_write_guard`` turns the guard on; ``hooks.batch_write_guard_commands``
    adds the operator's own write-command prefixes to the shell table. Off, with no
    extras, when config cannot be read.
    """
    try:
        from kiro_crew import hooks as hooks_mod
        from kiro_crew.config import KiroCrewConfig

        config = hooks_mod.HooksConfig.from_dict(KiroCrewConfig.load().hooks)
        return config.batch_write_guard, tuple(config.batch_write_guard_commands)
    except Exception:
        logger.debug("batch guard: config unreadable; guard off", exc_info=True)
        return False, ()


async def screen_approval(transport: Any, event: Any) -> str | None:
    """The guard's refusal for approving *event* on *transport*, or ``None``.

    When it refuses, the reason has already been steered into the running turn,
    so the caller only audits and rejects. The synchronous window check runs
    first; config is read only for a call that shares its batch with a read. The
    call itself is classified with the operator's extra write commands; its peers
    were classified without them when they were dispatched.
    """
    import asyncio

    window = getattr(transport, "_batch_window", None)
    if window is None or event is None:
        return None
    call_id = str(getattr(event, "tool_call_id", "") or "")
    if not window.read_peers(call_id):
        return None
    enabled, extra_write_commands = await asyncio.to_thread(guard_settings)
    if not enabled:
        return None
    reason = window.refusal_for(call_id, classify(event, extra_write_commands))
    if reason is None:
        return None
    from kiro_crew.deny_notice import steer_refusal_notice

    try:
        await steer_refusal_notice(
            transport,
            str(getattr(event, "title", "") or ""),
            reason,
            cause=DENY_CAUSE_BATCH_READ_WRITE,
        )
    except Exception:
        logger.debug("batch guard: notice steer failed; the reject still runs", exc_info=True)
    return reason
