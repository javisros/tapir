"""pytest if installed; otherwise a minimal fallback (parametrize, raises) and a runner."""
import contextlib, sys, types
try:
    import pytest
    FALLBACK = False
except ImportError:
    FALLBACK = True

    def _param(names, values):
        def deco(f):
            f._params = [v if isinstance(v, tuple) else (v,) for v in values]; return f
        return deco

    @contextlib.contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")

    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=_param), raises=_raises)


def main(path, namespace):
    if not FALLBACK:
        sys.exit(pytest.main([path, "-q"]))
    fails = 0
    for name, f in list(namespace.items()):
        if not name.startswith("test_"):
            continue
        for args in getattr(f, "_params", [()]):
            try:
                f(*args); print("ok  ", name, *args)
            except AssertionError as e:
                fails += 1; print("FAIL", name, *args, str(e)[:300])
    print("failures:", fails); sys.exit(1 if fails else 0)
