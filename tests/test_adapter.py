"""Unit tests for the mffpy adapter layer."""

import pytest
import numpy as np
from pathlib import Path

# try to find a test file
TEST_FILE = None
try:
    import mne
    egi_path = Path(mne.datasets.testing.data_path()) / 'EGI'
    mff_files = list(egi_path.glob('*.mff'))
    if mff_files:
        TEST_FILE = mff_files[0]
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    TEST_FILE is None, reason="No MFF test file available"
)


class TestMFFFileInfo:
    """Test metadata extraction."""

    def test_sfreq_positive(self):
        from src.adapter import MFFFileInfo
        info = MFFFileInfo(TEST_FILE)
        assert info.sfreq > 0

    def test_has_channels(self):
        from src.adapter import MFFFileInfo
        info = MFFFileInfo(TEST_FILE)
        assert info.n_channels > 0

    def test_channel_names_length(self):
        from src.adapter import MFFFileInfo
        info = MFFFileInfo(TEST_FILE)
        assert len(info.ch_names) == info.n_channels

    def test_segments_add_up(self):
        from src.adapter import MFFFileInfo
        info = MFFFileInfo(TEST_FILE)
        assert sum(info.segment_lengths) == info.n_samples

    def test_samples_positive(self):
        from src.adapter import MFFFileInfo
        info = MFFFileInfo(TEST_FILE)
        assert info.n_samples > 0


class TestReadRawData:
    """Test signal data reading."""

    def test_shape(self):
        from src.adapter import MFFFileInfo, read_raw_data
        info = MFFFileInfo(TEST_FILE)
        n = min(500, info.n_samples)
        data = read_raw_data(TEST_FILE, 0, n)
        assert data.shape == (info.n_channels, n)

    def test_finite_values(self):
        from src.adapter import read_raw_data
        data = read_raw_data(TEST_FILE, 0, 100)
        assert np.all(np.isfinite(data))

    def test_single_sample(self):
        from src.adapter import read_raw_data
        data = read_raw_data(TEST_FILE, 42, 43)
        assert data.shape[1] == 1

    def test_channel_picks(self):
        from src.adapter import read_raw_data
        picks = [0, 2, 5]
        data = read_raw_data(TEST_FILE, 0, 100, picks=picks)
        assert data.shape[0] == 3


class TestReadEvents:
    """Test event extraction."""

    def test_returns_correct_format(self):
        from src.adapter import MFFFileInfo, read_events
        info = MFFFileInfo(TEST_FILE)
        events, event_id = read_events(TEST_FILE, info.sfreq)
        assert events.ndim == 2
        if len(events) > 0:
            assert events.shape[1] == 3
            assert events.dtype == np.int64

    def test_event_ids_consistent(self):
        from src.adapter import MFFFileInfo, read_events
        info = MFFFileInfo(TEST_FILE)
        events, event_id = read_events(TEST_FILE, info.sfreq)
        if len(events) > 0:
            ids_in_array = set(events[:, 2])
            ids_in_dict = set(event_id.values())
            assert ids_in_array.issubset(ids_in_dict)


class TestEdgeCases:
    """Test error handling."""

    def test_bad_path(self):
        from src.adapter import MFFFileInfo
        with pytest.raises(FileNotFoundError):
            MFFFileInfo('/nonexistent/path.mff')

    def test_not_directory(self):
        from src.adapter import MFFFileInfo
        with pytest.raises(IOError):
            MFFFileInfo(__file__)