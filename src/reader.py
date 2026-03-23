# src/real_reader.py — bonus file showing BaseRaw integration

"""
Minimal but REAL BaseRaw subclass.
This actually plugs into MNE's infrastructure.
"""

import numpy as np
from pathlib import Path
import mne
from mne.io import BaseRaw
from mne.io.meas_info import create_info
from mne.io.constants import FIFF
from .adapter import MFFFileInfo, read_raw_data, read_events


class RawMffReal(BaseRaw):
    """Actual BaseRaw subclass — works with MNE's full pipeline."""
    
    def __init__(self, input_fname, preload=False, 
                 channel_naming='E%d', verbose=None):
        
        input_fname = Path(input_fname)
        
        # parse metadata through adapter
        self._mff_info = MFFFileInfo(input_fname)
        self._mff_path = input_fname
        
        # build channel names
        ch_names = [
            channel_naming % (i + 1) 
            for i in range(self._mff_info.n_channels)
        ]
        if self._mff_info.n_channels in (257, 129):
            ch_names[-1] = 'Cz'
        
        ch_types = ['eeg'] * self._mff_info.n_channels
        
        # create proper MNE Info object
        info = create_info(
            ch_names=ch_names,
            sfreq=self._mff_info.sfreq,
            ch_types=ch_types
        )
        
        # set calibrations to 1.0 — mffpy handles conversion
        for ch in info['chs']:
            ch['cal'] = 1.0
            ch['unit'] = FIFF.FIFF_UNIT_V
            ch['unit_mul'] = FIFF.FIFF_UNITM_NONE
        
        # set measurement date
        if self._mff_info.meas_date is not None:
            info.set_meas_date(self._mff_info.meas_date)
        
        # segment boundaries
        first_samps = np.array(
            self._mff_info.segment_starts, dtype=np.int64
        )
        last_samps = np.array([
            s + l - 1 for s, l in 
            zip(self._mff_info.segment_starts,
                self._mff_info.segment_lengths)
        ], dtype=np.int64)
        
        # call BaseRaw constructor
        super().__init__(
            info,
            preload=preload,
            first_samps=first_samps,
            last_samps=last_samps,
            filenames=[str(input_fname)] * self._mff_info.n_segments,
            orig_format='float',
            verbose=verbose,
        )
        
        # add events as annotations
        events, event_id = read_events(input_fname, self._mff_info.sfreq)
        if len(events) > 0:
            onsets = events[:, 0] / self._mff_info.sfreq
            durations = np.zeros(len(events))
            inv_id = {v: k for k, v in event_id.items()}
            descriptions = [inv_id.get(e, str(e)) for e in events[:, 2]]
            
            annots = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=self._mff_info.meas_date
            )
            self.set_annotations(annots)
    
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """BaseRaw calls this for lazy loading.
        
        We read only the requested chunk through our adapter.
        """
        offset = self._mff_info.segment_starts[fi]
        
        # read through adapter — only the samples we need
        chunk = read_raw_data(
            self._mff_path, 
            offset + start, 
            offset + stop
        )
        
        # select requested channels
        chunk = chunk[idx]
        
        # apply calibration and projectors
        # this is the standard BaseRaw pattern
        chunk *= cals[idx][:, np.newaxis]
        
        if mult is not None:
            data[:] += mult @ chunk
        else:
            data[:] += chunk