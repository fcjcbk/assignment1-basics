from __future__ import annotations

import argparse

from cs336_basics.cli import add_generate_arguments, run_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a saved Transformer checkpoint.")
    add_generate_arguments(parser)
    return parser.parse_args()


def main() -> None:
    run_generate(parse_args())


if __name__ == "__main__":
    main()
