"""Operational CLI: bootstrap an admin key, inspect the stack.

Deliberately small. The ETL has its own CLI (``app.etl.cli``) because it has a
genuinely different surface; this one exists so that a fresh `docker compose up`
can be made usable without anybody writing SQL by hand.
"""

from __future__ import annotations

import logging

import typer

from app.config import get_api_settings

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Narrative Intelligence backend operations.")


@app.command("bootstrap-key")
def bootstrap_key(
    name: str = typer.Option("bootstrap-admin", help="Label for the key."),
    scopes: str = typer.Option("read,write,admin", help="Comma-separated scopes."),
    force: bool = typer.Option(False, "--force", help="Mint even if active keys already exist."),
) -> None:
    """Mint the first admin key.

    Refuses by default once any active key exists. A command that silently mints
    a fresh admin credential every time somebody runs it is how a system ends up
    with fourteen live admin keys and no idea who holds them.
    """
    from sqlalchemy import select

    from app.db import sync_session
    from app.models.ops import ApiKey
    from app.security import mint

    settings = get_api_settings()
    if not settings.api_key_pepper:
        typer.secho(
            "API_KEY_PEPPER is unset. Generate one with\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            "and put it in .env before minting a key.",
            fg="red",
        )
        raise typer.Exit(2)

    with sync_session() as session:
        existing = (
            session.execute(select(ApiKey).where(ApiKey.revoked_at.is_(None))).scalars().all()
        )
        if existing and not force:
            typer.secho(
                f"{len(existing)} active key(s) already exist "
                f"({', '.join(k.prefix for k in existing[:5])}). "
                "Use POST /api/v1/keys with an admin key, or pass --force.",
                fg="yellow",
            )
            raise typer.Exit(1)

        minted = mint()
        session.add(
            ApiKey(
                name=name,
                key_hash=minted.key_hash,
                prefix=minted.prefix,
                scopes=[s.strip() for s in scopes.split(",") if s.strip()],
            )
        )

    typer.secho("\nAPI key minted. This is the only time it will be shown:\n", fg="green")
    typer.echo(f"  {minted.plaintext}\n")
    typer.echo(f"  curl -H 'X-API-Key: {minted.plaintext}' http://localhost:8000/api/v1/projects\n")


@app.command("config")
def show_config() -> None:
    """Print the resolved settings, with secrets redacted."""
    settings = get_api_settings()
    redact = {"api_key_pepper", "bootstrap_admin_key", "anthropic_api_key"}
    for field in sorted(settings.model_fields):
        value = getattr(settings, field)
        if field in redact:
            value = "<set>" if value else "<unset>"
        elif "url" in field and isinstance(value, str) and "@" in value:
            # Never print a password, not even to a developer's terminal that
            # may well be inside a screen recording.
            scheme, _, rest = value.partition("://")
            value = f"{scheme}://***@{rest.split('@', 1)[1]}"
        typer.echo(f"{field:32} {value}")


@app.command("capabilities")
def capabilities() -> None:
    """Report which model capabilities this deployment has."""
    from nlp.availability import probe

    for capability in probe(refresh=True):
        colour = {"ready": "green", "degraded": "yellow", "unavailable": "red"}[capability.status]
        typer.secho(f"{capability.name:12} {capability.status:12}", fg=colour, nl=False)
        typer.echo(capability.detail)


def main() -> None:  # pragma: no cover - entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
