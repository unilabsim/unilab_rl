def test_import():
    import uni_rl

    assert uni_rl.__version__


def test_no_unilab_dependency():
    """uni_rl must never import unilab (decoupling contract)."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "uni_rl"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import unilab", "from unilab")):
                offenders.append(f"{path}:{line.strip()}")
    assert not offenders, "uni_rl imports unilab:\n" + "\n".join(offenders)
