"""
Adapter layer between mffpy and MNE-Python.

This module handles ALL interaction with mffpy. Nothing else in the
project imports mffpy directly. This isolation means if mffpy changes
its API tomorrow, we only fix this one file.

Key features:
- Lazy loading: read_raw_data() reads ONLY the requested samples
- ISO 8601 date parsing: proper timezone-aware measurement dates
- Epoch stitching: handles data requests that span segment boundaries
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timezone

try:
    import mffpy
    HAS_MFFPY = True
except ImportError:
    HAS_MFFPY = False


def check_mffpy():
    """Raise a clear error if mffpy is not installed."""
    if not HAS_MFFPY:
        raise ImportError(
            "mffpy is required to read MFF files. "
            "Install with: pip install mffpy"
        )


def _parse_mff_date(date_value):
    """Parse measurement date into timezone-aware datetime.

    EGI MFF files store dates in ISO 8601 format like:
        2024-03-15T14:30:00.000000+05:30
        2024-03-15T09:00:00.000000Z

    MNE-Python expects timezone-aware datetime objects in UTC.
    If we get this wrong, the recording shows up as 1970-01-01
    which is obviously broken.

    Parameters
    ----------
    date_value : str or datetime or None
        Date from mffpy reader.

    Returns
    -------
    dt : datetime (UTC, timezone-aware) or None
    """
    if date_value is None:
        return None

    # already a datetime object
    if isinstance(date_value, datetime):
        if date_value.tzinfo is None:
            # assume UTC if no timezone
            return date_value.replace(tzinfo=timezone.utc)
        return date_value

    # string — parse ISO 8601
    if isinstance(date_value, str):
        try:
            # python 3.7+ handles most ISO formats
            dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

        # fallback: try common EGI format
        for fmt in [
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
        ]:
            try:
                dt = datetime.strptime(date_value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

    # couldn't parse — return None rather than crashing
    return None


class MFFFileInfo:
    """Parse all metadata from an MFF file through mffpy.

    This replaces ~300 lines of XML parsing in MNE's _parse_mff.py
    with about 60 lines that just ask mffpy for the answers.

    Parameters
    ----------
    mff_path : str or Path
        Path to the .mff directory.

    Attributes
    ----------
    sfreq : float
        Sampling frequency in Hz.
    n_channels : int
        Number of EEG channels.
    ch_names : list of str
        Channel names.
    n_samples : int
        Total sample count across all segments.
    n_segments : int
        Number of segments (epochs).
    segment_starts : list of int
        Starting sample index of each segment.
    segment_lengths : list of int
        Number of samples in each segment.
    meas_date : datetime or None
        Recording date (UTC, timezone-aware).
    pns_n_channels : int
        Number of PNS (physiology) channels.
    pns_ch_names : list of str
        PNS channel names.
    pns_sfreq : float or None
        PNS sampling frequency.
    """

    def __init__(self, mff_path):
        check_mffpy()
        self.path = Path(mff_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")
        if not self.path.is_dir():
            raise IOError(f"MFF path must be a directory: {self.path}")

        reader = mffpy.Reader(str(self.path))
        reader.set_unit('EEG', 'V')

        # --- EEG info ---
        eeg_blocks = [b for b in reader.blocks
                      if b['signal_type'] == 'EEG']
        if not eeg_blocks:
            raise RuntimeError(f"No EEG data in {self.path}")

        self.sfreq = float(eeg_blocks[0]['sampling_rate'])
        self.n_channels = eeg_blocks[0]['num_channels']
        self.ch_names = self._build_channel_names()

        # --- Segments ---
        self.segment_starts = []
        self.segment_lengths = []
        total = 0
        for ep in reader.epochs:
            self.segment_starts.append(total)
            n = int(ep['num_samples'])
            self.segment_lengths.append(n)
            total += n

        self.n_samples = total
        self.n_segments = len(self.segment_starts)

        # --- Measurement date (ISO 8601) ---
        raw_date = None
        try:
            raw_date = reader.startdatetime
        except Exception:
            pass
        self.meas_date = _parse_mff_date(raw_date)

        # --- PNS channels ---
        pns_blocks = [b for b in reader.blocks
                      if b.get('signal_type') == 'PNS']
        if pns_blocks:
            self.pns_sfreq = float(pns_blocks[0]['sampling_rate'])
            self.pns_n_channels = pns_blocks[0]['num_channels']
            self.pns_ch_names = [f'PNS{i + 1}'
                                 for i in range(self.pns_n_channels)]
        else:
            self.pns_sfreq = None
            self.pns_n_channels = 0
            self.pns_ch_names = []

    def _build_channel_names(self):
        names = [f'E{i + 1}' for i in range(self.n_channels)]
        if self.n_channels in (257, 129):
            names[-1] = 'Cz'
        return names

    def get_segment_for_sample(self, sample):
        """Find which segment contains a given sample index.

        Used by lazy loading to figure out which epoch to read.

        Returns
        -------
        seg_idx : int
            Segment index.
        local_sample : int
            Sample index within that segment.
        """
        for i, (start, length) in enumerate(
                zip(self.segment_starts, self.segment_lengths)):
            if start <= sample < start + length:
                return i, sample - start
        raise IndexError(f"Sample {sample} is out of range "
                         f"(max: {self.n_samples})")


def read_raw_data(mff_path, start, stop, signal_type='EEG', picks=None):
    """Read ONLY the requested samples from disk.

    This is the lazy loading function. If a file is 10GB and you ask
    for 1 second of data, this reads only that 1 second. It doesn't
    load the whole file into memory.

    It handles the tricky case where your requested range spans
    multiple epochs by reading pieces from each epoch and stitching
    them together.

    Parameters
    ----------
    mff_path : str or Path
        Path to .mff directory.
    start : int
        First sample (inclusive).
    stop : int
        Last sample (exclusive).
    signal_type : str
        'EEG' or 'PNS'.
    picks : list of int or None
        Channel indices to read. None = all channels.

    Returns
    -------
    data : ndarray, shape (n_channels, stop - start)
        Signal data in volts (physical units).
    """
    check_mffpy()

    # create a fresh reader each time — thread safe
    # (important for MNE's n_jobs parallel processing)
    reader = mffpy.Reader(str(mff_path))
    reader.set_unit(signal_type, 'V')

    pieces = []
    running = 0
    bytes_read = 0

    for epoch in reader.epochs:
        ep_len = int(epoch['num_samples'])
        ep_end = running + ep_len

        # skip epochs entirely before our range
        if ep_end <= start:
            running = ep_end
            continue
        # stop once we've passed our range
        if running >= stop:
            break

        # read this epoch from disk
        ep_data = reader.get_physical_samples_from_epoch(epoch)
        block = ep_data[signal_type]

        # calculate which slice of this epoch we actually need
        local_start = max(0, start - running)
        local_stop = min(ep_len, stop - running)

        # .copy() is critical here — without it, the slice holds
        # a reference to the full epoch array and python won't
        # free the memory
        piece = block[:, local_start:local_stop].copy()
        pieces.append(piece)

        bytes_read += piece.nbytes

        # immediately free the full epoch from memory
        del ep_data, block

        running = ep_end

    if not pieces:
        raise RuntimeError(
            f"No data for range [{start}, {stop}) in {mff_path}"
        )

    data = np.concatenate(pieces, axis=1)

    if picks is not None:
        data = data[np.array(picks)]

    return data


def read_events(mff_path, sfreq):
    """Read events from MFF and convert to MNE format.

    Parameters
    ----------
    mff_path : str or Path
        Path to .mff directory.
    sfreq : float
        Sampling frequency for timestamp conversion.

    Returns
    -------
    events : ndarray, shape (n_events, 3)
        MNE-style [sample, 0, event_id].
    event_id : dict
        Mapping from event name to integer id.
    """
    check_mffpy()
    reader = mffpy.Reader(str(mff_path))

    events_list = []
    event_id = {}
    next_id = 1

    # try categories
    try:
        categories = reader.categories
        if categories:
            for name, content in categories.items():
                if name not in event_id:
                    event_id[name] = next_id
                    next_id += 1
                eid = event_id[name]
                for seg in content:
                    onset_us = seg.get('beginTime', 0)
                    sample = int(round(onset_us * sfreq / 1e6))
                    events_list.append([sample, 0, eid])
    except Exception:
        pass

    # try events track
    try:
        raw_events = reader.events
        if raw_events:
            for ev in raw_events:
                code = ev.get('code', 'unknown')
                if code not in event_id:
                    event_id[code] = next_id
                    next_id += 1
                eid = event_id[code]
                onset_us = ev.get('beginTime', 0)
                sample = int(round(onset_us * sfreq / 1e6))
                events_list.append([sample, 0, eid])
    except Exception:
        pass

    if events_list:
        events = np.array(
            sorted(events_list, key=lambda x: x[0]),
            dtype=np.int64
        )
    else:
        events = np.empty((0, 3), dtype=np.int64)

    return events, event_id