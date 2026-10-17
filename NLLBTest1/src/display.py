"""Shared terminal presentation: banner, status line and the `:command` language."""
import json
import re
from pathlib import Path

from src.config import ALIASES, DIRECTIONS, LANG_NAMES, SHORT_NAMES

QUIT_COMMANDS = {"q", "quit", ":q", ":quit", "exit"}
MENU_COMMANDS = {"m", "menu", ":m", ":menu"}
HELP_COMMANDS = {"?", "h", "help", ":h", ":help"}
BEAM_CHOICES = (1, 4, 8)

HELP_LINES = (
    "just type a sentence and press Enter",
    "  :1..:4  pick a direction        :swap   flip the current direction",
    "  :dirs   list directions        :beams N  beam search (1, 4, 8)",
    "  :save <file>  dump the session :clear    empty the screen",
    "  :help   this list              :q       quit",
)
KEY_HINTS = "Enter translate | Ctrl+R swap | Ctrl+L clear | Ctrl+S save | Ctrl+C quit"


def short(language):
    return SHORT_NAMES.get(language, language[:3])


def pick_language(key):
    """Map a typed word onto a language: exact alias first ('achagua'), then prefix ('piapo')."""
    key = key.strip().lower()
    for language in LANG_NAMES:
        if key in {alias for alias, name in ALIASES.items() if name == language}:
            return language
    for language in LANG_NAMES:
        aliases = {alias for alias, name in ALIASES.items() if name == language and len(alias) >= 3}
        if any(key.startswith(alias) for alias in aliases):
            return language
    return None


def arrow(source, target):
    return f"{short(source)}->{short(target)}"


def training_summary(model_dir):
    report = Path(model_dir) / "training_report.json"
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def format_banner(model_dir, tags, mode, device=None, beams=None):
    lines = ["NLLB translator  Spanish <-> Piapoco / Achagua",
             f"  adapter : {model_dir}",
             f"  tags    : " + ", ".join(f"{name}={tag}" for name, tag in tags.as_dict().items())]
    summary = training_summary(model_dir)
    if summary:
        rows = ", ".join(f"{name}: {count}" for name, count in summary.get("rows", {}).items())
        lines.append(f"  data    : {rows or 'unknown'} pairs ({summary.get('tag_mode', mode)} tags)")
        for language, info in summary.get("report", {}).items():
            scores = "  ".join(f"{direction} {value['chrf_pp']}"
                               for direction, value in info["directions"].items())
            lines.append(f"  chrF++  : {language:8s} {scores}  [{info['evaluated_on']}]")
    runtime = [f"  device  : {device}"] if device else []
    if beams is not None:
        runtime.append(f"beams: {beams}")
    if runtime:
        lines.append("".join(runtime).rstrip())
    return "\n".join(lines)


def format_status(source, target, device=None, beams=None, message=None):
    parts = [f" {arrow(source, target)} "]
    if device:
        parts.append(f" {device}")
    if beams:
        parts.append(f" beams {beams}")
    if message:
        parts.append(f"  {message}")
    return "".join(parts)


def format_menu(directions=DIRECTIONS, active=None):
    lines = ["directions:"]
    for index, (source, target) in enumerate(directions, start=1):
        marker = "*" if active == (source, target) else " "
        lines.append(f"  {marker} {index}) {short(source)} -> {short(target)}")
    return "\n".join(lines)


def parse_command(text, directions=DIRECTIONS):
    """Turn a typed line into (kind, value) so both UIs behave the same."""
    stripped = text.strip()
    if not stripped:
        return "empty", None
    if stripped.lower() in QUIT_COMMANDS:
        return "quit", None
    if stripped.lower() in MENU_COMMANDS:
        return "menu", None
    if stripped.lower() in HELP_COMMANDS:
        return "help", None
    if not stripped.startswith(":"):
        return "translate", stripped

    head, _, rest = stripped[1:].partition(" ")
    name, rest = head.lower(), rest.strip()
    if name in ("q", "quit", "exit"):
        return "quit", None
    if name in ("m", "menu", "dirs", "directions"):
        return "menu", None
    if name in ("h", "help", "?"):
        return "help", None
    if name in ("1", "2", "3", "4") and not rest:
        index = int(name) - 1
        if 0 <= index < len(directions):
            return "direction", directions[index]
        return "error", f"there is no direction {name}"
    if name in ("d", "dir", "lang") and rest:
        return "pick", rest
    if name == "swap":
        return "swap", None
    if name == "clear":
        return "clear", None
    if name in ("save", "s"):
        return ("save", rest) if rest else ("error", "usage: :save <file>")
    if name == "beams":
        try:
            value = int(rest)
        except ValueError:
            return "error", "usage: :beams 4"
        return ("beams", value) if value in BEAM_CHOICES else ("error", f"beams must be one of {BEAM_CHOICES}")
    return "error", f"unknown command :{name} (:help)"


def resolve_direction(spec, current, directions=DIRECTIONS):
    """Accept '1', '2', 'achagua', 'es->aca' and 'swap'."""
    key = spec.strip().lower()
    if key in ("swap", "reverse", "flip"):
        return (current[1], current[0]) if current in directions else directions[0]
    if key.isdigit():
        index = int(key) - 1
        return directions[index] if 0 <= index < len(directions) else None
    if "->" in key or ">" in key:
        left, right = re.split(r"->|>", key, maxsplit=1)
        pair = (pick_language(left), pick_language(right))
        return next((direction for direction in directions if direction == pair), None)
    wanted = pick_language(key)
    if wanted is None:
        return None
    return (next((d for d in directions if d[0] == current[0] and d[1] == wanted), None)
            or next((d for d in directions if d[1] == wanted), None)
            or next((d for d in directions if d[0] == wanted), None))