def test_import():
    import uni_rl

    assert uni_rl.__version__


def test_no_unilab_dependency():
    """uni_rl must never import unilab (decoupling contract).

    TEMPORARY (issue #1478): migrated modules still import unilab.base.*,
    unilab.utils.{sim2sim,checkpoint} and unisim.* until the decoupling in
    issue #1479 lands. The gate is therefore relaxed to "no unmarked unilab
    imports": every offender import statement must carry the
    `# TODO(issue-1479)` marker. Restore the strict `assert not offenders`
    once #1479 completes.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "uni_rl"
    offenders = []
    unmarked = []
    for path in src.rglob("*.py"):
        lines = path.read_text().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith(("import unilab", "from unilab")):
                # A parenthesized import may span several lines; ruff/isort can
                # move the trailing marker onto an inner symbol line, so scan
                # the whole statement for it.
                block = [line]
                if stripped.endswith("("):
                    while not lines[i].strip().startswith(")"):
                        i += 1
                        block.append(lines[i])
                text = "\n".join(block)
                offenders.append(f"{path}:{stripped}")
                if "TODO(issue-1479)" not in text:
                    unmarked.append(f"{path}:{stripped}")
            i += 1
    # Surfaced for visibility while the temporary imports exist.
    if offenders:
        print("unilab imports pending #1479 decoupling:\n" + "\n".join(offenders))
    assert not unmarked, "uni_rl has unmarked unilab imports:\n" + "\n".join(unmarked)
