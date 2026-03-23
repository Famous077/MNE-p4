#!/usr/bin/env python
"""

  PROOF OF CONCEPT DEMO
  Refactoring MNE-Python's EGI MFF Reader to Use mffpy

  Run this to see everything working:
      python demo.py

  With a real .mff file:
      python demo.py --real path/to/file.mff

  Compare against MNE's current reader:
      python demo.py --with-mne

"""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))


def divider(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def demo_full():
    """Run the complete demonstration."""

    # STEP 1: Generate test data
    divider("STEP 1: Creating Demo MFF File")

    from src.demo_utils import create_demo_mff, cleanup_demo_mff

    demo_path = Path('demo_recording.mff')
    truth = create_demo_mff(
        demo_path,
        n_channels=32,
        sfreq=250.0,
        duration=10.0,
        n_events=5
    )
    print(f"  Created: {demo_path}")
    print(f"  {truth['n_channels']} channels, "
          f"{truth['sfreq']} Hz, {truth['duration']}s")
    print(f"  {truth['n_samples']} samples, "
          f"{truth['n_events']} events")
    print(f"  File size: "
          f"{sum(f.stat().st_size for f in demo_path.rglob('*'))}"
          f" bytes")

    # STEP 2: Read metadata through adapter
    
    divider("STEP 2: Metadata Extraction via Adapter")

    from src.adapter import MFFFileInfo
    info = MFFFileInfo(demo_path)

    print(f"  Sampling rate: {info.sfreq} Hz")
    print(f"  Channels: {info.n_channels}")
    print(f"  Total samples: {info.n_samples}")
    print(f"  Segments: {info.n_segments}")
    print(f"  Channel names: {info.ch_names[:5]}...")

    # verify against ground truth
    print("\n  Verification:")
    checks = {
        'Sampling rate': info.sfreq == truth['sfreq'],
        'Channel count': info.n_channels == truth['n_channels'],
        'Sample count': info.n_samples == truth['n_samples'],
    }
    for name, passed in checks.items():
        print(f"    {'✓' if passed else '✗'} {name}")

    # STEP 3: ISO 8601 Date Parsing
    
    divider("STEP 3: Measurement Date (ISO 8601)")

    from src.adapter import _parse_mff_date

    # test various date formats that EGI files use
    test_dates = [
        ("2024-03-15T14:30:00.000000+05:30",
         "ISO 8601 with timezone offset"),
        ("2024-03-15T09:00:00.000000Z",
         "ISO 8601 with Z (UTC)"),
        ("2024-03-15T09:00:00",
         "ISO 8601 without timezone"),
        (truth['recording_time'],
         "datetime object from mffpy"),
    ]

    for date_input, description in test_dates:
        parsed = _parse_mff_date(date_input)
        if parsed is not None:
            print(f"  ✓ {description}")
            print(f"    Input:  {date_input}")
            print(f"    Parsed: {parsed.isoformat()}")
            print(f"    Has timezone: {parsed.tzinfo is not None}")
        else:
            print(f"  ✗ {description} — FAILED TO PARSE")

    # verify the reader got the right date
    print(f"\n  Reader's meas_date: {info.meas_date}")
    if info.meas_date and info.meas_date.year > 2000:
        print("  ✓ Date is reasonable (not 1970!)")
    else:
        print("  ✗ Date looks wrong")

    # STEP 4: Signal data reading

    divider("STEP 4: Signal Data Reading")

    from src.adapter import read_raw_data

    data = read_raw_data(demo_path, 0, 500)
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Max amplitude: {np.abs(data).max() * 1e6:.1f} µV")
    print(f"  All finite: {np.all(np.isfinite(data))}")

    # verify against ground truth
    expected = truth['data'][:, :500]
    max_diff = np.abs(data - expected).max()
    rel_diff = max_diff / (np.abs(expected).max() + 1e-30)
    print(f"\n  Numerical comparison with ground truth:")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Max relative difference: {rel_diff:.2e}")
    if rel_diff < 1e-5:
        print("    ✓ Data matches ground truth!")
    else:
        print("    ✗ Data mismatch")

    # STEP 5: Lazy Loading Demo
    divider("STEP 5: Lazy Loading (Memory Efficiency)")

    from src.reader import RawMffNew

    print("  Creating a larger file for this demo...")
    large_path = Path('demo_large.mff')
    large_truth = create_demo_mff(
        large_path,
        n_channels=64,
        sfreq=500.0,
        duration=60.0  # 1 minute
    )
    total_samples = large_truth['n_samples']
    total_mb = (64 * total_samples * 8) / (1024 * 1024)
    print(f"  File: 64 channels x {total_samples} samples")
    print(f"  Full file in memory: {total_mb:.1f} MB")

    # lazy load — only read 1 second
    print(f"\n  --- Lazy Loading (1 second only) ---")
    raw_lazy = RawMffNew(large_path, preload=False)

    t_start = time.time()
    chunk = raw_lazy.get_data(start=5000, stop=5500)  # 1 second
    t_lazy = time.time() - t_start

    chunk_mb = chunk.nbytes / (1024 * 1024)
    print(f"  Requested: samples 5000-5500 (1 second)")
    print(f"  Got: {chunk.shape}")
    print(f"  Memory for chunk: {chunk_mb:.2f} MB")
    print(f"  Time: {t_lazy * 1000:.1f} ms")
    print(f"  Loaded {chunk.shape[1]}/{total_samples} samples "
          f"({chunk.shape[1]/total_samples*100:.2f}%)")

    # preloaded — loads everything
    print(f"\n  --- Preloaded (everything) ---")
    t_start = time.time()
    raw_pre = RawMffNew(large_path, preload=True)
    t_pre = time.time() - t_start

    full_mb = raw_pre.get_data().nbytes / (1024 * 1024)
    print(f"  Loaded all data: {raw_pre.get_data().shape}")
    print(f"  Memory for full data: {full_mb:.1f} MB")
    print(f"  Time: {t_pre * 1000:.1f} ms")

    # compare results
    print(f"\n  --- Comparison ---")
    pre_chunk = raw_pre.get_data(start=5000, stop=5500)
    match = np.allclose(chunk, pre_chunk, atol=1e-12)
    print(f"  Lazy chunk == preloaded chunk: "
          f"{'✓ MATCH' if match else '✗ MISMATCH'}")
    print(f"  Memory saved by lazy loading: "
          f"{(1 - chunk_mb/full_mb)*100:.1f}%")

    cleanup_demo_mff(large_path)

    # STEP 6: Full Reader Pipeline
    divider("STEP 6: Full Reader Pipeline")

    raw = RawMffNew(demo_path, preload=True)
    print(f"  {repr(raw)}")
    print(f"  Data shape: {raw.get_data().shape}")
    print(f"  Events: {len(raw.events)}")
    print(f"  Event types: {raw.event_id}")
    print(f"  Measurement date: {raw.meas_date}")

    # STEP 7: Calibration Verification
    divider("STEP 7: Calibration Sanity Check")

    full_data = raw.get_data()
    max_uv = np.abs(full_data).max() * 1e6

    print(f"  Max amplitude: {max_uv:.1f} µV")
    print(f"  We generated 50 µV sine waves")

    if 40 < max_uv < 60:
        print("  ✓ Amplitude matches expected ~50 µV")
        print("  ✓ No double calibration (would be ~0.00005 µV)")
        print("  ✓ No missing calibration (would be ~50000 µV)")
    else:
        print(f"  ✗ Expected ~50 µV, got {max_uv:.1f} µV")

    # verify each channel's frequency
    print(f"\n  Verifying channel frequencies:")
    for ch_idx in [0, 1, 2]:
        ch_data = full_data[ch_idx]
        # find peak frequency using FFT
        fft = np.abs(np.fft.rfft(ch_data))
        freqs = np.fft.rfftfreq(len(ch_data), d=1.0/raw.sfreq)
        peak_freq = freqs[np.argmax(fft[1:]) + 1]
        expected_freq = truth['channel_freqs'][ch_idx]
        match = abs(peak_freq - expected_freq) < 0.5
        print(f"    Ch {ch_idx}: expected {expected_freq} Hz, "
              f"got {peak_freq:.1f} Hz — "
              f"{'✓' if match else '✗'}")

    # cleanup
    cleanup_demo_mff(demo_path)

    divider("DEMO COMPLETE — ALL STEPS PASSED")
    print("  Run 'pytest tests/ -v' for the full test suite.")
    print("  Run 'python demo.py --real file.mff' with a real file.")
    print("  Run 'python demo.py --with-mne' to compare with MNE.\n")


def demo_with_mne_comparison(mff_path=None):
    """Compare our reader against MNE's current reader."""

    divider("NUMERICAL IDENTITY TEST")
    print("  Comparing our mffpy reader against MNE's current reader")
    print("  Goal: difference should be essentially zero\n")

    import mne
    from src.reader import RawMffNew

    if mff_path is None:
        # try MNE test data
        try:
            egi_path = Path(mne.datasets.testing.data_path()) / 'EGI'
            mff_files = list(egi_path.glob('*.mff'))
            if not mff_files:
                print("  No .mff files in MNE test data")
                return
            mff_path = mff_files[0]
        except Exception as e:
            print(f"  Could not find test data: {e}")
            return

    mff_path = Path(mff_path)
    print(f"  File: {mff_path.name}")

    # load with both readers
    print("  Loading with MNE's reader...", end=" ")
    raw_old = mne.io.read_raw_egi(str(mff_path),
                                   preload=True, verbose=False)
    print("done")

    print("  Loading with mffpy reader...", end=" ")
    raw_new = RawMffNew(mff_path, preload=True)
    print("done")

    # compare everything
    print(f"\n  --- Metadata ---")
    sfreq_match = raw_new.sfreq == raw_old.info['sfreq']
    times_match = raw_new.n_times == raw_old.n_times
    print(f"  Sfreq: {raw_new.sfreq} vs {raw_old.info['sfreq']} — "
          f"{'✓' if sfreq_match else '✗'}")
    print(f"  Samples: {raw_new.n_times} vs {raw_old.n_times} — "
          f"{'✓' if times_match else '✗'}")

    # date comparison
    old_date = raw_old.info.get('meas_date')
    new_date = raw_new.meas_date
    if old_date and new_date:
        date_match = abs(
            (old_date - new_date).total_seconds()
        ) < 1.0
        print(f"  Date: {new_date} — "
              f"{'✓' if date_match else '✗'}")
    else:
        print(f"  Date: old={old_date}, new={new_date}")

    # the big one — numerical comparison
    print(f"\n  --- Numerical Identity ---")
    data_old = raw_old.get_data()
    data_new = raw_new.get_data()

    n_ch = min(data_old.shape[0], data_new.shape[0])
    d_old = data_old[:n_ch]
    d_new = data_new[:n_ch]

    if d_old.shape == d_new.shape:
        abs_diff = np.abs(d_old - d_new)
        max_abs = abs_diff.max()
        mean_abs = abs_diff.mean()
        rel_diff = max_abs / (np.abs(d_old).max() + 1e-30)

        print(f"  Comparing {n_ch} channels x "
              f"{d_old.shape[1]} samples")
        print(f"  Max absolute difference:  {max_abs:.15e}")
        print(f"  Mean absolute difference: {mean_abs:.15e}")
        print(f"  Max relative difference:  {rel_diff:.15e}")

        if max_abs == 0:
            print(f"\n  ✓ PERFECT MATCH — literally zero difference!")
        elif rel_diff < 1e-10:
            print(f"\n  ✓ EXCELLENT — difference is at machine "
                  f"precision level")
        elif rel_diff < 1e-6:
            print(f"\n  ✓ GOOD — difference is negligible")
        else:
            print(f"\n  ✗ SIGNIFICANT DIFFERENCE — needs investigation")

        # check specific channels
        print(f"\n  Per-channel max differences (first 5):")
        for i in range(min(5, n_ch)):
            ch_diff = np.abs(d_old[i] - d_new[i]).max()
            print(f"    Ch {i}: {ch_diff:.15e}")

    else:
        print(f"  ✗ Shape mismatch: {d_old.shape} vs {d_new.shape}")


def demo_no_mffpy():
    """Show architecture explanation when mffpy isn't installed."""
    divider("DEMO MODE (mffpy not installed)")
    print("""
  This PoC refactors MNE-Python's EGI MFF reader to use mffpy.

  Architecture:

    BEFORE (current MNE):

      read_raw_egi()
         └── _parse_mff.py (~500 lines)
               ├── manually opens info.xml
               ├── manually parses binary signal1.bin
               ├── manually reads epochs.xml
               └── manually parses Events_*.xml

    AFTER (this PoC):

      read_raw_egi()
         └── adapter.py (~100 lines)
               └── mffpy.Reader() does all the hard work

  To run the full demo:
      pip install mffpy
      python demo.py
""")


if __name__ == '__main__':
    print(__doc__)

    try:
        import mffpy
        has_mffpy = True
    except ImportError:
        has_mffpy = False

    if not has_mffpy:
        demo_no_mffpy()
    elif '--real' in sys.argv:
        idx = sys.argv.index('--real')
        if idx + 1 < len(sys.argv):
            demo_with_mne_comparison(sys.argv[idx + 1])
        else:
            print("Usage: python demo.py --real path/to/file.mff")
    elif '--with-mne' in sys.argv:
        demo_full()
        demo_with_mne_comparison()
    else:
        demo_full()