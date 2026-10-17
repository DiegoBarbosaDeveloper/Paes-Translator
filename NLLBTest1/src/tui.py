"""Full-screen terminal UI for the NLLB translator, built on prompt_toolkit."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from src.config import DIRECTIONS, NUM_BEAMS
from src.display import (BEAM_CHOICES, KEY_HINTS, arrow, format_banner, format_menu, parse_command,
                         resolve_direction, short)
from src.files import write_output
from src.modeling import LanguageTags, Translator

STYLE = Style.from_dict({
    "header": "bold ansicyan",
    "rule": "fg:#606060",
    "banner": "fg:#909090",
    "label": "bold ansimagenta",
    "translation": "bold ansigreen",
    "error": "ansired",
    "note": "fg:#909090",
    "status": "fg:#707070",
    "prompt": "bold ansicyan",
    "entry": "",
    "answer": "bold ansigreen",
})

RULE = "─" * 100
SCROLL_STEP = 12
ARROWS = tuple(f"{short(source)}->{short(target)}" for source, target in DIRECTIONS)


class TranscriptLexer(Lexer):
    """Colours the transcript by line prefix: direction label, answer, error or plain note."""

    def lex_document(self, document):
        def line_at(lineno):
            line = document.lines[lineno]
            stripped = line.lstrip()
            for arrow in ARROWS:
                if stripped.startswith(arrow):
                    end = line.index(arrow) + len(arrow)
                    return [("class:label", line[:end]), ("class:entry", line[end:])]
            if stripped.startswith("->"):
                return [("class:answer", line)]
            if stripped.startswith(("error", "!")):
                return [("class:error", line)]
            return [("class:note", line)]

        return line_at


class TranslatorTui:
    """Holds all interface state; prompt_toolkit renders from it and calls back into it."""

    def __init__(self, model_dir, num_beams=NUM_BEAMS, translator=None):
        self.model_dir = model_dir
        self.num_beams = num_beams
        self.direction = DIRECTIONS[0]
        self.translator = translator
        self.device = str(translator.device) if translator is not None else None
        self.message = ""
        self.busy = False
        self.log = []
        self.lock = threading.Lock()
        self.inflight = set()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.tags, self.mode = LanguageTags.load(model_dir)
        # The transcript is a read-only buffer so prompt_toolkit keeps the newest line in view
        # and handles scrolling for us; every append moves the cursor to the end.
        self.transcript = Buffer(multiline=False)
        # `multiline=False` is required: the default `multiline=True` makes prompt_toolkit insert a
        # newline on Enter instead of calling the accept handler.
        self.input_buffer = Buffer(multiline=False, accept_handler=self.submit)
        # Built inside `run()` because Application() resolves the terminal session at construction time.
        self.application = None
        self.banner_shown = False

    # ---------- transcript ----------

    def write(self, text=""):
        with self.lock:
            current = self.transcript.text
            self.transcript.document = Document(text=current + text + "\n",
                                                cursor_position=len(current) + len(text) + 1)

    def scroll(self, delta):
        document = self.transcript.document
        row = max(0, min(len(document.lines) - 1, document.cursor_position_row + delta))
        position = document.translate_row_column_to_index(row, 0)
        self.transcript.document = Document(text=document.text, cursor_position=position)
        self.application.invalidate()

    # ---------- chrome ----------

    def header_text(self):
        tags = "  ".join(f"{name}={tag}" for name, tag in self.tags.as_dict().items())
        return [("class:header", " NLLB translator   Spanish <-> Piapoco / Achagua "),
                ("class:note", f"  {tags}")]

    def rule_text(self):
        return [("class:rule", RULE)]

    def status_text(self):
        device = self.device or "model loads on the first translation"
        if self.busy:
            return [("class:status", f" {device}  translating...")]
        beams = "" if self.num_beams == 1 else f"beams {self.num_beams}  "
        return [("class:status", f" {device}  {beams}{self.message or KEY_HINTS}")]

    def direction_text(self):
        return [("class:label", f" {arrow(*self.direction)} "), ("class:prompt", "❯ ")]

    def build_layout(self):
        return Layout(HSplit([
            Window(FormattedTextControl(self.header_text), height=1),
            Window(FormattedTextControl(self.rule_text), height=1),
            Window(BufferControl(self.transcript, lexer=TranscriptLexer(), focusable=False), wrap_lines=True),
            Window(FormattedTextControl(self.rule_text), height=1),
            VSplit([
                Window(FormattedTextControl(self.direction_text), width=13, height=1),
                Window(BufferControl(self.input_buffer), height=1),
            ]),
            Window(FormattedTextControl(self.status_text), height=1),
        ]), focused_element=self.input_buffer)

    # ---------- model ----------

    def ensure_translator(self):
        if self.translator is None:
            self.translator = Translator.load(self.model_dir)
            self.device = str(self.translator.device)
        return self.translator

    def request(self, text, source, target, beams):
        self.busy = True
        self.message = ""
        self.write(f" {arrow(source, target)}  {text}")

        def work():
            return self.ensure_translator().translate_one(text, source, target, num_beams=beams)

        future = self.executor.submit(work)
        with self.lock:
            self.inflight.add(future)
        future.add_done_callback(lambda done: self.settle(done, text, source, target))
        return future

    def settle(self, future, text, source, target):
        """Hop back onto the event loop so every state change happens on the UI thread.

        The future leaves `inflight` here, not in `finish()`: otherwise a `:save` that waits on the
        queue would block the very loop that has to run `finish()` to release it.
        """
        with self.lock:
            self.inflight.discard(future)
        try:
            translation = future.result()
        except Exception as error:
            translation, failure = None, error
        else:
            failure = None
            # Recorded here rather than in finish() so a waiting :save sees a complete log.
            with self.lock:
                self.log.append((text, translation))
        loop = self.application.loop if self.application is not None else None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self.finish, text, source, target, translation, failure)
        else:
            self.finish(text, source, target, translation, failure)

    def finish(self, text, source, target, translation, failure):
        self.busy = False
        if failure is not None:
            self.write(f"    error: {failure}")
            self.message = "failed"
        else:
            self.write(f"    -> {translation}")
            self.message = "ok"
        self.application.invalidate()

    def snapshot(self):
        """A consistent copy of the session, safe to call while a translation is in flight."""
        with self.lock:
            return list(self.log)

    def wait_for_translations(self, timeout=None):
        """Block until the queue is empty, so a save or quit cannot lose an in-flight line."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.inflight:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            with self.lock:
                pending = list(self.inflight)
            if not pending:
                break
            try:
                pending[0].result(timeout=remaining)
            except Exception:
                break
        # Let the queued callbacks render before the caller (a save or quit) continues.
        loop = self.application.loop if self.application is not None else None
        if loop is not None and loop.is_running():
            self.application.invalidate()

    # ---------- commands ----------

    def submit(self, buffer):
        text = buffer.text.strip()
        buffer.reset()
        if not text:
            return
        kind, value = parse_command(text)

        if kind == "translate":
            self.request(value, *self.direction, beams=self.num_beams)
        elif kind in ("direction", "swap"):
            self.direction = resolve_direction("swap", self.direction) if kind == "swap" else value
            self.message = f"direction {arrow(*self.direction)}"
        elif kind == "pick":
            resolved = resolve_direction(value, self.direction)
            if resolved is None:
                self.write(f" error: unknown direction {value!r} (:help)")
                self.message = "unknown direction"
            else:
                self.direction = resolved
                self.message = f"direction {arrow(*resolved)}"
        elif kind == "beams":
            self.num_beams = value
            self.message = f"beam search {value}"
        elif kind == "clear":
            self.wait_for_translations()
            self.transcript.document = Document(text="\n")
            self.log.clear()
            self.show_banner()
            self.message = "cleared"
        elif kind == "save":
            self.wait_for_translations()
            entries = self.snapshot()
            path = write_output([item[0] for item in entries], [item[1] for item in entries], value)
            self.write(f" saved {len(entries)} lines to {path}")
            self.message = f"saved {path.name}"
        elif kind == "help":
            self.show_help()
        elif kind == "quit":
            # A raised exception inside a key handler is swallowed by prompt_toolkit, so exit cleanly.
            self.wait_for_translations()
            get_app().exit()
        elif kind == "error":
            self.write(f" error: {value}")
            self.message = value
        self.application.invalidate()

    def show_banner(self):
        self.write()
        self.write(format_banner(self.model_dir, self.tags, self.mode,
                                  device=self.device, beams=self.num_beams))
        self.write("type a sentence and press Enter   |   :help for commands   |   :q to quit")
        self.banner_shown = True

    def show_help(self):
        self.write()
        self.write(format_menu(active=self.direction))
        for line in KEY_HINTS.split("|"):
            self.write(f"   {line.strip()}")
        self.message = "help"

    # ---------- keys ----------

    def build_key_bindings(self):
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _(event):
            if event.current_buffer.text:
                event.current_buffer.reset()
                event.app.invalidate()
            else:
                event.app.exit()

        @bindings.add("c-d")
        @bindings.add("c-q")
        def _(event):
            # Wait for pending translations so their transcript lines still get rendered.
            event.app.exit()

        @bindings.add("c-r")
        def _(event):
            self.direction = resolve_direction("swap", self.direction)
            self.message = f"direction {arrow(*self.direction)}"
            event.app.invalidate()

        @bindings.add("c-l")
        def _(event):
            self.wait_for_translations()
            self.transcript.document = Document(text="\n")
            self.log.clear()
            self.show_banner()
            self.message = "cleared"
            event.app.invalidate()

        @bindings.add("c-s")
        def _(event):
            event.current_buffer.text = ":save translations.csv"
            event.current_buffer.cursor_position = len(":save translations.csv")
            event.app.invalidate()

        @bindings.add("f1")
        def _(event):
            self.show_help()
            event.app.invalidate()

        @bindings.add("c-up")
        def _(event):
            self.scroll(-SCROLL_STEP)

        @bindings.add("c-down")
        def _(event):
            self.scroll(SCROLL_STEP)

        @bindings.add("escape")
        def _(event):
            event.current_buffer.reset()
            event.app.invalidate()

        return bindings

    def run(self):
        self.transcript.document = Document(text="\n")
        self.show_banner()
        self.application = Application(
            layout=self.build_layout(),
            key_bindings=self.build_key_bindings(),
            style=STYLE,
            full_screen=True,
        )
        try:
            self.application.run()
        finally:
            self.wait_for_translations()
            self.executor.shutdown(wait=False, cancel_futures=True)
            entries = self.snapshot()
            print(f"session: {len(entries)} translations" + ("  (not saved)" if entries else ""))
        return entries


def run_tui(model_dir, num_beams=NUM_BEAMS, translator=None, beam_choices=BEAM_CHOICES):
    return TranslatorTui(model_dir, num_beams=num_beams, translator=translator).run()