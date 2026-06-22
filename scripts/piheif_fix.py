"""
piheif_fix.py
-------------
Silence the noisy "corrupt image / pi_heif" warnings that Ultralytics + Pillow
emit when scanning a dataset. The HTW image set sometimes triggers a probe for
the optional `pi_heif` package; when it is missing OR when Pillow's image
verification stumbles on a non-HEIF file, a flood of warnings appears.

Fix strategy:
  1. Register a *fake* `pi_heif` module in sys.modules so the optional-import
     probe in Pillow/ultralytics succeeds quietly (its register_heif_opener()
     becomes a no-op).
  2. Patch pkg_resources so a version lookup for "pi-heif" does not raise
     DistributionNotFound (older code paths call get_distribution()).

Import this module ONCE at the top of every pipeline notebook BEFORE importing
ultralytics / PIL:

    import sys; sys.path.insert(0, "/home/jovyan/shared/s0598584/scripts")
    import piheif_fix  # noqa: F401
"""
import sys
import types
import warnings


def _install_fake_pi_heif():
    if "pi_heif" in sys.modules:
        return
    fake = types.ModuleType("pi_heif")

    def register_heif_opener(*args, **kwargs):  # no-op
        return None

    fake.register_heif_opener = register_heif_opener
    fake.__version__ = "0.0.0-fake"
    sys.modules["pi_heif"] = fake
    # some code imports the C-style name
    sys.modules["pillow_heif"] = fake


def _patch_pkg_resources():
    try:
        import pkg_resources
    except Exception:
        return
    _orig_get = pkg_resources.get_distribution

    class _FakeDist:
        version = "0.0.0-fake"

    def _safe_get(name, *a, **k):
        try:
            return _orig_get(name, *a, **k)
        except Exception:
            if "heif" in str(name).lower():
                return _FakeDist()
            raise

    pkg_resources.get_distribution = _safe_get


def apply():
    _install_fake_pi_heif()
    _patch_pkg_resources()
    # also mute the generic "Corrupt EXIF data" / "Possibly corrupt" PIL warnings
    warnings.filterwarnings("ignore", message=".*Corrupt.*")
    warnings.filterwarnings("ignore", message=".*Possibly corrupt.*")


# auto-apply on import
apply()
