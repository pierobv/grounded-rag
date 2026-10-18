import pytest

from grounded_rag.cli import build_parser


def test_ask_parses_question():
    args = build_parser().parse_args(["ask", "O que é XSS?"])

    assert args.command == "ask"
    assert args.question == "O que é XSS?"


def test_unknown_command_exits_with_error():
    # argparse encerra o programa (SystemExit) com código 2 em erro de uso.
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["voar"])

    assert exc.value.code == 2
