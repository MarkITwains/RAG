"""Export the installed, resolved WSL dependency closure without local paths.

Run after installing .[runtime,ollama,dev], then verify a clean install with
the resulting constraints. This is an environment-specific version snapshot;
it deliberately does not claim wheel hashes or local-model compatibility.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def resolved_versions(requirements):
    environment = default_environment()
    seen, versions = set(), {}

    def visit(spec, parent_extras=("",)):
        requirement = Requirement(spec)
        if requirement.marker and not any(
            requirement.marker.evaluate(dict(environment, extra=extra)) for extra in parent_extras
        ):
            return
        name = canonicalize_name(requirement.name)
        key = name, tuple(sorted(requirement.extras))
        if key in seen:
            return
        seen.add(key)
        distribution = metadata.distribution(name)
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            raise ValueError(f"installed {name}=={distribution.version} violates {requirement.specifier}")
        versions[name] = distribution.version
        for dependency in distribution.requires or ():
            visit(dependency, ("", *requirement.extras))

    for requirement in requirements:
        visit(requirement)
    versions.pop("pcb-rag", None)
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("requirements/constraints-wsl-py312.txt"))
    args = parser.parse_args(argv)
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        parser.error("this snapshot targets Linux/WSL Python 3.12")
    import tomllib

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = list(config["project"].get("dependencies", []))
    extras = ("runtime", "ollama", "dev")
    for extra in extras:
        requirements.extend(config["project"]["optional-dependencies"][extra])
    requirements.extend(config["build-system"]["requires"])
    versions = resolved_versions(requirements)
    header = (
        "# Resolved version snapshot: WSL/Linux x86_64, Python " + platform.python_version() + "\n"
        "# Profile: .[runtime,ollama,dev] plus the build backend.\n"
        "# Regenerate: python scripts/export_wsl_constraints.py\n"
        "# Install: python -m pip install -c requirements/constraints-wsl-py312.txt -e '.[runtime,ollama,dev]'\n"
        "# Local torch models and model weights are not included.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + "".join(f"{key}=={versions[key]}\n" for key in sorted(versions)), encoding="utf-8")
    print(f"Wrote {len(versions)} package versions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
