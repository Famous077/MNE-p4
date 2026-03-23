"""
Creates valid .mff files for demonstration purposes.

Uses mffpy's Writer class to generate properly formatted MFF files
that mffpy's Reader can then read back. This means our demo works
without needing any real EEG recordings.

The generated data contains sine waves at known frequencies and
amplitudes so we can verify the reader produces correct numbers.
"""

import numpy as np
import shutil
from pathlib import Path
from datetime import datetime, timezone


def create_demo_mff(output_path, n_channels=32, sfreq=250.0,
                    duration=10.0, n_events=5):
    """Create a valid .mff file with known synthetic data.

    Uses mffpy.Writer to generate a properly formatted MFF directory
    that any MFF reader (mffpy or MNE's current reader) can open.

    Parameters
    ----------
    output_path : str or Path
        Where to create the .mff directory.
    n_channels : int
        Number of EEG channels to generate.
    sfreq : float
        Sampling rate in Hz.
    duration : float
        Recording length in seconds.
    n_events : int
        Number of event markers to insert.

    Returns
    -------
    ground_truth : dict
        Contains the exact data we wrote so we can verify the
        reader gets the same numbers back.
    """
    try:
        from mffpy import Writer, BinWriter
    except ImportError:
        raise ImportError(
            "mffpy is required to create demo files. "
            "Install with: pip install mffpy"
        )

    path = Path(output_path)
    if path.exists():
        shutil.rmtree(path)

    n_samples = int(sfreq * duration)
    recording_time = datetime.now(tz=timezone.utc)

    # --- generate known signal data ---
    # each channel gets a unique sine wave so we can verify
    # channel ordering is preserved
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples), dtype=np.float32)

    for i in range(n_channels):
        freq = 1.0 + i * 0.5  # ch0: 1Hz, ch1: 1.5Hz, ch2: 2Hz...
        amplitude = 50e-6      # 50 microvolts (realistic EEG)
        data[i] = amplitude * np.sin(2 * np.pi * freq * t)

    # --- write using mffpy ---
    writer = Writer(str(path))
    writer.addentry(recording_time)

    # create binary writer for EEG data
    bin_writer = BinWriter(
        sampling_rate=int(sfreq),
        data_type='EEG'
    )
    bin_writer.add_block(data)
    writer.add_signal(bin_writer)

    # write to disk
    writer.write()

    # --- generate event timestamps ---
    event_times = np.linspace(1.0, duration - 1.0, n_events)
    event_samples = np.round(event_times * sfreq).astype(int)

    # --- return ground truth ---
    ground_truth = {
        'n_channels': n_channels,
        'sfreq': sfreq,
        'n_samples': n_samples,
        'duration': duration,
        'n_events': n_events,
        'data': data.astype(np.float64),  # float64 for comparison
        'event_times': event_times,
        'event_samples': event_samples,
        'recording_time': recording_time,
        'path': path,
        'channel_freqs': [1.0 + i * 0.5 for i in range(n_channels)],
    }

    return ground_truth


def create_large_demo_mff(output_path, n_channels=256, sfreq=1000.0,
                          duration=300.0):
    """Create a larger .mff file for memory/lazy loading tests.

    5 minutes of 256-channel data at 1000Hz = ~580MB
    This is big enough to show that lazy loading matters.
    """
    return create_demo_mff(
        output_path,
        n_channels=n_channels,
        sfreq=sfreq,
        duration=duration,
        n_events=20
    )


def cleanup_demo_mff(path):
    """Remove a demo .mff directory."""
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)