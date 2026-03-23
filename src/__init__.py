"""
Proof of Concept: MNE-Python EGI MFF Reader Refactoring with mffpy.

Import chain (how files connect):

    demo.py
      └── src.reader.RawMffNew
            └── src.adapter.MFFFileInfo
            └── src.adapter.read_raw_data  (lazy loading happens here)
            └── src.adapter.read_events
                  └── mffpy.Reader  (external library)
                        └── .mff files on disk

    demo.py also uses:
      └── src.demo_utils.create_demo_mff  (generates test data)
            └── mffpy.Writer  (creates valid .mff files)
"""

from .adapter import MFFFileInfo, read_raw_data, read_events, check_mffpy
from .reader import RawMffNew
from .demo_utils import create_demo_mff, cleanup_demo_mff

__all__ = [
    'MFFFileInfo',
    'read_raw_data',
    'read_events',
    'check_mffpy',
    'RawMffNew',
    'create_demo_mff',
    'cleanup_demo_mff',
]