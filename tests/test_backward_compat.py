"""Tests that work WITHOUT real MFF files.

These generate fake data so they run anywhere — on CI, on a
mentor's laptop, on a fresh install. No downloads needed.
"""

import pytest
import numpy as np
from pathlib import Path

mffpy = pytest.importorskip('mffpy')

DEMO_PATH = Path('test_temp.mff')


@pytest.fixture(autouse=True)
def cleanup():
    """Remove temp files after each test."""
    yield
    from src.demo_utils import cleanup_demo_mff
    cleanup_demo_mff(DEMO_PATH)


class TestDemoFileCreation:

    def test_creates_valid_mff(self):
        from src.demo_utils import create_demo_mff
        truth = create_demo_mff(DEMO_PATH, n_channels=8, duration=2.0)
        assert DEMO_PATH.is_dir()
        # verify mffpy can read it
        reader = mffpy.Reader(str(DEMO_PATH))
        assert len(reader.epochs) > 0

    def test_ground_truth_shape(self):
        from src.demo_utils import create_demo_mff
        truth = create_demo_mff(
            DEMO_PATH, n_channels=16, sfreq=500.0, duration=5.0
        )
        assert truth['data'].shape == (16, 2500)
        assert truth['sfreq'] == 500.0


class TestAdapterWithFakeData:

    def test_metadata(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import MFFFileInfo
        truth = create_demo_mff(DEMO_PATH, n_channels=8, sfreq=250.0)
        info = MFFFileInfo(DEMO_PATH)
        assert info.sfreq == 250.0
        assert info.n_channels == 8
        assert info.n_samples == truth['n_samples']

    def test_data_shape(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import read_raw_data
        create_demo_mff(DEMO_PATH, n_channels=8, duration=2.0)
        data = read_raw_data(DEMO_PATH, 0, 100)
        assert data.shape == (8, 100)

    def test_data_finite(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import read_raw_data
        create_demo_mff(DEMO_PATH, n_channels=4)
        data = read_raw_data(DEMO_PATH, 0, 200)
        assert np.all(np.isfinite(data))

    def test_data_matches_truth(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import read_raw_data
        truth = create_demo_mff(DEMO_PATH, n_channels=4, duration=1.0)
        data = read_raw_data(DEMO_PATH, 0, truth['n_samples'])
        np.testing.assert_allclose(data, truth['data'], rtol=1e-5)

    def test_single_sample(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import read_raw_data
        create_demo_mff(DEMO_PATH, n_channels=4)
        data = read_raw_data(DEMO_PATH, 42, 43)
        assert data.shape == (4, 1)

    def test_channel_picks(self):
        from src.demo_utils import create_demo_mff
        from src.adapter import read_raw_data
        create_demo_mff(DEMO_PATH, n_channels=16)
        data = read_raw_data(DEMO_PATH, 0, 50, picks=[0, 3, 7])
        assert data.shape == (3, 50)


class TestDateParsing:

    def test_iso_with_timezone(self):
        from src.adapter import _parse_mff_date
        dt = _parse_mff_date("2024-03-15T14:30:00.000000+05:30")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.year == 2024

    def test_iso_with_z(self):
        from src.adapter import _parse_mff_date
        dt = _parse_mff_date("2024-03-15T09:00:00.000000Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_iso_without_timezone(self):
        from src.adapter import _parse_mff_date
        dt = _parse_mff_date("2024-03-15T09:00:00")
        assert dt is not None
        # should default to UTC
        assert dt.tzinfo is not None

    def test_none_input(self):
        from src.adapter import _parse_mff_date
        assert _parse_mff_date(None) is None

    def test_datetime_object(self):
        from src.adapter import _parse_mff_date
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        result = _parse_mff_date(now)
        assert result == now

    def test_not_1970(self):
        """This catches the common bug where dates default to epoch."""
        from src.demo_utils import create_demo_mff
        from src.adapter import MFFFileInfo
        truth = create_demo_mff(DEMO_PATH, n_channels=4)
        info = MFFFileInfo(DEMO_PATH)
        if info.meas_date is not None:
            assert info.meas_date.year > 2000, \
                "Date is 1970 — ISO 8601 parsing is broken!"


class TestLazyLoading:

    def test_lazy_reads_less_data(self):
        """Lazy loading should not load the entire file."""
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew

        create_demo_mff(DEMO_PATH, n_channels=32, duration=30.0)

        # lazy — only read 1 second
        raw = RawMffNew(DEMO_PATH, preload=False)
        assert raw._data is None  # nothing loaded yet
        chunk = raw.get_data(start=0, stop=250)
        assert chunk.shape == (32, 250)
        assert raw._data is None  # still nothing fully loaded

    def test_lazy_matches_preloaded(self):
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew

        create_demo_mff(DEMO_PATH, n_channels=8, duration=5.0)

        raw_lazy = RawMffNew(DEMO_PATH, preload=False)
        raw_pre = RawMffNew(DEMO_PATH, preload=True)

        lazy_chunk = raw_lazy.get_data(start=100, stop=600)
        pre_chunk = raw_pre.get_data(start=100, stop=600)

        np.testing.assert_allclose(lazy_chunk, pre_chunk, atol=1e-12)

    def test_lazy_middle_of_file(self):
        """Read from the middle without loading start or end."""
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew

        create_demo_mff(DEMO_PATH, n_channels=8, sfreq=250.0,
                        duration=10.0)
        raw = RawMffNew(DEMO_PATH, preload=False)

        # read 1 second from the middle (samples 1250-1500)
        middle = raw.get_data(start=1250, stop=1500)
        assert middle.shape == (8, 250)
        assert np.all(np.isfinite(middle))


class TestCalibration:

    def test_amplitude_range(self):
        """Data should be in realistic microvolt range."""
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew

        create_demo_mff(DEMO_PATH, n_channels=8)
        raw = RawMffNew(DEMO_PATH, preload=True)
        max_uv = np.abs(raw.get_data()).max() * 1e6

        # we generated 50 µV signals
        assert 10 < max_uv < 200, \
            f"Expected ~50 µV, got {max_uv:.1f} µV"

    def test_no_double_calibration(self):
        """Catch the double-calibration bug."""
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew

        create_demo_mff(DEMO_PATH, n_channels=4)
        raw = RawMffNew(DEMO_PATH, preload=True)
        max_amp = np.abs(raw.get_data()).max()

        # double cal would give ~1e-11, missing cal ~1e3
        assert max_amp > 1e-8, \
            f"Amplitude {max_amp} suggests double calibration"
        assert max_amp < 1e-2, \
            f"Amplitude {max_amp} suggests missing calibration"


class TestEdgeCases:

    def test_bad_path(self):
        from src.adapter import MFFFileInfo
        with pytest.raises(FileNotFoundError):
            MFFFileInfo('/this/does/not/exist.mff')

    def test_not_directory(self):
        from src.adapter import MFFFileInfo
        with pytest.raises(IOError):
            MFFFileInfo(__file__)

    def test_repr(self):
        from src.demo_utils import create_demo_mff
        from src.reader import RawMffNew
        create_demo_mff(DEMO_PATH, n_channels=4, duration=2.0)
        raw = RawMffNew(DEMO_PATH, preload=True)
        text = repr(raw)
        assert 'RawMffNew' in text
        assert '.mff' in text