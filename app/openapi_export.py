"""Write ``openapi.json`` from the code, or fail if the committed copy drifted.

The committed schema is a deliverable, not a build artifact: it is the contract
Phase 5 builds against, and a change to it is a change to the frontend's world.
Regenerating it in CI and diffing means a route rename cannot land without
somebody noticing that it breaks the dashboard.

    make openapi          # regenerate
    make openapi-check    # fail on drift
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_schema() -> dict:
    """Generate the schema with DEMO_MODE forced off.

    The schema must not depend on runtime state. Generating it with DEMO_MODE=1
    would produce the same shapes today, but tying the published contract to an
    environment variable is exactly the kind of coupling that eventually bites.
    """
    import os

    os.environ["DEMO_MODE"] = "0"
    from app.config import get_api_settings

    get_api_settings.cache_clear()

    from app.main import create_app

    return create_app().openapi()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    target = Path(argv[0] if argv else "openapi.json")

    schema = build_schema()
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if check:
        if not target.exists():
            print(f"{target} does not exist. Run `make openapi`.", file=sys.stderr)
            return 1
        if target.read_text(encoding="utf-8") != rendered:
            print(
                f"{target} has drifted from the code.\n"
                "The OpenAPI schema is the frontend contract; regenerate it with "
                "`make openapi` and review the diff before committing.",
                file=sys.stderr,
            )
            return 1
        print(f"{target} is up to date ({len(schema['paths'])} paths).")
        return 0

    target.write_text(rendered, encoding="utf-8")
    operations = sum(
        1
        for path in schema["paths"].values()
        for method in path
        if method in {"get", "post", "patch", "put", "delete"}
    )
    print(f"wrote {target}: {len(schema['paths'])} paths, {operations} operations")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
