"""File handoff uses precisely the same service and canonical bytes as /v2."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from api.contracts.generation_v1 import canonical_bytes
from api.generation_service import GenerationService


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "command", choices=["capabilities", "plan", "approve", "run", "status", "result", "cancel"]
    )
    parser.add_argument("input", nargs="?")
    args = parser.parse_args(argv)
    # Provider progress belongs on stderr; stdout is one canonical document.
    with redirect_stdout(sys.stderr):
        config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
        svc = GenerationService(args.root, config)
        if args.command in {"plan", "approve"}:
            result = getattr(svc, args.command)(json.loads(Path(args.input).read_text()))
        elif args.command == "capabilities":
            result = svc.capabilities()
        else:
            result = getattr(svc, "job" if args.command == "status" else args.command)(args.input)
    sys.stdout.buffer.write(canonical_bytes(result))


if __name__ == "__main__":
    main()
