"""Translate with the fine-tuned NLLB model: full-screen TUI, plain prompt, or one-shot flags."""
import argparse
import os
import sys

import sacrebleu

from src.config import ADAPTER_DIR, DIRECTIONS, EVAL_BATCH_SIZE, NUM_BEAMS
from src.console import force_utf8
from src.display import (BEAM_CHOICES, HELP_LINES, arrow, format_banner, format_menu, format_status,
                         parse_command, resolve_direction)
from src.files import read_lines, write_output
from src.modeling import LanguageTags, Translator, resolve_language

UI_CHOICES = ("auto", "tui", "plain")


class Quit(Exception):
    """Raised when a plain-mode session ends (EOF, Ctrl+C or :q)."""


def print_banner(translator, model_dir, num_beams=None):
    tags, mode = LanguageTags.load(model_dir)
    device = str(translator.device) if translator is not None else None
    print(format_banner(model_dir, tags, mode, device=device, beams=num_beams))


def print_loaded(translator):
    print(f"model loaded on {translator.device} ({translator.num_beams} beams)")


def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit from None


def is_interactive_terminal():
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def choose_ui(requested):
    """`auto` upgrades to the full-screen TUI only when a real terminal is attached."""
    forced = os.environ.get("NLLB_UI", "").strip().lower()
    if forced in UI_CHOICES:
        requested = forced
    if requested == "auto":
        return "tui" if is_interactive_terminal() else "plain"
    return requested


def load_inputs(args):
    """Collect (texts, reference) from the flags, or return ([], None) for interactive mode."""
    texts, reference = [], None
    if args.text:
        texts.extend(args.text)
    if args.csv_column or args.csv_reference:
        if not args.file:
            raise SystemExit("--csv-column/--csv-reference need --file")
        from_file, reference = read_lines(args.file, args.csv_column, args.csv_reference)
        texts.extend(from_file)
    elif args.file:
        from_file, _ = read_lines(args.file)
        texts.extend(from_file)
    if args.stdin:
        texts.extend(line.strip() for line in sys.stdin if line.strip())
    return texts, reference


def run_once(translator, args):
    texts, reference = load_inputs(args)
    if not texts:
        return False
    source = resolve_language(args.src)
    target = resolve_language(args.tgt)
    translations = translator.translate(texts, source, target, num_beams=args.num_beams,
                                        batch_size=args.batch_size)
    for text, translation in zip(texts, translations):
        print(f"{text}\n  -> {translation}")
    if reference:
        score = sacrebleu.corpus_chrf(translations, [reference], word_order=2).score
        print(f"\nchrF++ = {score:.1f}")
    if args.output:
        write_output(texts, translations, args.output, reference)
        print(f"wrote {args.output}")
    return True


def show_help():
    print("\ncommands:")
    for line in HELP_LINES:
        print(f"  {line}")
    print()


def run_interactive(translator, model_dir, num_beams=NUM_BEAMS):
    """Line based fallback for notebooks, pipes and terminals that cannot host a full-screen UI."""
    direction = DIRECTIONS[0]
    beams = num_beams
    log = []
    print_banner(translator, model_dir, beams)
    print("Type a sentence and press Enter. Commands: " + "  ".join(HELP_LINES[1:5]))
    try:
        while True:
            text = ask(format_status(*direction, beams=beams) + " > ")
            if not text:
                continue
            kind, value = parse_command(text)

            if kind == "translate":
                if translator is None:
                    print("loading the model, first run downloads NLLB (~2.5 GB) ...")
                    translator = Translator.load(model_dir)
                    print_loaded(translator)
                source, target = direction
                translation = translator.translate_one(value, source, target, num_beams=beams)
                print(f"  -> {translation}")
                log.append((value, translation))
            elif kind in ("direction", "swap"):
                direction = resolve_direction("swap", direction) if kind == "swap" else value
            elif kind == "pick":
                resolved = resolve_direction(value, direction)
                direction = resolved or direction
                if resolved is None:
                    print(f"unknown direction {value!r}, try :menu")
            elif kind == "beams":
                beams = value
                print(f"beam search = {beams}")
            elif kind == "menu":
                print("\n" + format_menu(active=direction))
            elif kind == "help":
                show_help()
            elif kind == "clear":
                log.clear()
                print("(log cleared)")
            elif kind == "save":
                write_output([item[0] for item in log], [item[1] for item in log], value)
                print(f"saved {len(log)} lines to {value}")
            elif kind == "quit":
                break
            elif kind == "error":
                print(value)
    except Quit:
        pass
    if log:
        print(f"({len(log)} translations in this session, use ':save <file>' next time)")
    print("bye")


def run_tui(model_dir, num_beams=NUM_BEAMS):
    try:
        from src.tui import run_tui as launch
    except ImportError as error:
        raise SystemExit(f"prompt_toolkit is required for the TUI ({error}); use --ui plain") from error
    try:
        launch(model_dir, num_beams=num_beams)
    except Exception as error:
        if not is_interactive_terminal():
            raise SystemExit(f"the TUI needs a real terminal ({error}); use --ui plain") from error
        raise


def build_parser():
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="Translate Spanish <-> Piapoco/Achagua with the fine-tuned NLLB model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="No arguments starts the interface (TUI in a terminal, plain prompts otherwise).\n"
               "Examples:\n"
               "  python -m src.cli --ui tui\n"
               "  python -m src.cli --ui plain --src spanish --tgt piapoco --text \"Buenos días\"\n"
               "  python -m src.cli --src achagua --tgt spanish --stdin < sentences.txt\n"
               "  python -m src.cli --src spanish --tgt achagua --file data/processed/spanish_achagua.csv\n"
               "      --csv-column spanish --csv-reference achagua --output predictions.csv",
    )
    parser.add_argument("--src", help="source language (es, piapoco, achagua)")
    parser.add_argument("--tgt", help="target language (es, piapoco, achagua)")
    parser.add_argument("--text", action="append", help="text to translate, repeatable")
    parser.add_argument("--file", help="input file, one sentence per line, or a CSV with --csv-column")
    parser.add_argument("--csv-column", help="column to read from --file when it is a CSV")
    parser.add_argument("--csv-reference", help="column with the expected translation, enables chrF++")
    parser.add_argument("--stdin", action="store_true", help="also read sentences from stdin")
    parser.add_argument("--output", help="write results here (.csv or .txt)")
    parser.add_argument("--model-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS,
                        help=f"beam search width ({', '.join(map(str, BEAM_CHOICES))})")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--ui", choices=UI_CHOICES, default="auto",
                        help="auto (default) uses the TUI on a terminal and plain prompts elsewhere")
    parser.add_argument("--list-directions", action="store_true", help="print the supported directions")
    return parser


def default_argv():
    """`sys.argv` is polluted by IPython, so a notebook call must fall back to no arguments."""
    if "IPython" in sys.modules or "ipykernel" in sys.modules:
        return []
    return sys.argv[1:]


def main(argv=None):
    force_utf8()
    args = build_parser().parse_args(default_argv() if argv is None else argv)

    if args.list_directions:
        for source, target in DIRECTIONS:
            print(f"{arrow(source, target)}  ({source} -> {target})")
        return

    if not args.stdin and not args.file and not args.text:
        if choose_ui(args.ui) == "tui":
            run_tui(args.model_dir, args.num_beams)
        else:
            run_interactive(None, args.model_dir, args.num_beams)
        return

    if not (args.src and args.tgt):
        raise SystemExit("--src and --tgt are required when you pass input")

    print_banner(None, args.model_dir, args.num_beams)
    translator = Translator.load(args.model_dir, num_beams=args.num_beams)
    print_loaded(translator)
    run_once(translator, args)


if __name__ == "__main__":
    main()