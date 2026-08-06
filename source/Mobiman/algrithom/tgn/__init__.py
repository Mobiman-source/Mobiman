"""
TGN package.

Re-exports TGNStreaming from the parent tgn.py file which is shadowed
by this package directory.
"""
import importlib.util
import os

# tgn.py lives alongside this tgn/ directory
_tgn_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tgn.py")

_spec = importlib.util.spec_from_file_location("_tgn_streaming_module", _tgn_py)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TGNStreaming = _mod.TGNStreaming
