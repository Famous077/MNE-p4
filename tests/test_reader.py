# tests/test_real_reader.py

"""Test the actual BaseRaw subclass."""

import pytest
import numpy as np

mffpy = pytest.importorskip('mffpy')
mne = pytest.importorskip('mne')


class TestRealReader:
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from src.demo_utils import create_demo_mff
        self.mff_path = tmp_path / 'test.mff'
        self.truth = create_demo_mff(
            self.mff_path, n_channels=8, duration=5.0
        )
    
    def test_is_base_raw(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=True)
        assert isinstance(raw, mne.io.BaseRaw)
    
    def test_filtering_works(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=True)
        # THIS is what the PoC couldn't do before
        raw.filter(1.0, 40.0)
    
    def test_epoching_works(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=True)
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, 
                           baseline=None, preload=True)
        assert len(epochs) > 0
    
    def test_plot_works(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=True)
        # just check it doesn't crash
        fig = raw.plot(show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_lazy_loading_via_baseraw(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=False)
        # BaseRaw's __getitem__ calls _read_segment_file
        data = raw[:, 100:200][0]
        assert data.shape == (8, 100)
        assert np.all(np.isfinite(data))
    
    def test_save_and_reload(self):
        from src.real_reader import RawMffReal
        raw = RawMffReal(self.mff_path, preload=True)
        
        # save to FIF format
        out_path = self.mff_path.parent / 'test_raw.fif'
        raw.save(out_path, overwrite=True)
        
        # reload and compare
        raw2 = mne.io.read_raw_fif(out_path, preload=True)
        np.testing.assert_allclose(
            raw.get_data(), raw2.get_data(), atol=1e-7
        )