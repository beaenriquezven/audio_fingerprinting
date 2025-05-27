import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from scipy.signal import butter, filtfilt

class Generator:
    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 4096,
                 hop_length: int = 128,
                 fan_value: int = 2,
                 max_peaks: int = 10,
                 amp_min_db: float = None,
                 zone_time: float = 1.0,
                 zone_freq: float = 600):
    
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fan_value = fan_value
        self.max_peaks = max_peaks
        self.amp_min_db = amp_min_db
        self.zone_time = zone_time
        self.zone_freq = zone_freq

    def compute_spectrogram(self, y: np.ndarray):
        nyq = self.sr / 2
        b, a = butter(4, [300/nyq, 3000/nyq], btype='band')
        y = filtfilt(b, a, y)

        # —————— CÁLCULO DEL ESPECTROGRAMA ——————
        freqs, times, Sxx = spectrogram(
            y, fs=self.sr, window='hann',
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            mode='magnitude'
        )
        Sxx = 10 * np.log10(Sxx + 1e-10)
        return freqs, times, Sxx

    def get_peaks(self, Sxx: np.ndarray, times: np.ndarray, freqs: np.ndarray):
        # detección de picos igual que antes
        neighborhood = maximum_filter(Sxx, size=(40,40))
        mask = (Sxx == neighborhood)
        mags = Sxx[mask]
        if mags.size:
            th = np.percentile(mags, 75)
            mask &= (Sxx > th)
        # Umbral adaptativo: media+1·std por columna (frame)
        if self.amp_min_db is None:
            col_means = Sxx.mean(axis=0)
            col_stds  = Sxx.std(axis=0)
            thresh   = col_means + col_stds
            mask &= (Sxx > thresh[np.newaxis, :])
        else:
            mask &= (Sxx > self.amp_min_db)

        peaks = np.argwhere(mask)
        if peaks.size == 0:
            return np.empty((0,4), dtype=int)  # devolver array vacío de forma consistente

        f_idxs, t_idxs = peaks[:,0], peaks[:,1]
            # Ahora extraemos hasta max_peaks Picos por COLUMNA (tiempo)
        selected = []
        for col in np.unique(t_idxs):
            mask_col = t_idxs == col
            idxs = np.where(mask_col)[0]
            mags = Sxx[f_idxs[idxs], t_idxs[idxs]]
            topk = idxs[np.argsort(mags)[-min(len(mags), self.max_peaks):]]
            selected.extend(topk)
        sel = np.array(selected, dtype=int)
        t_idxs = t_idxs[sel]
        f_idxs = f_idxs[sel]

        # construyo un array de 4 columnas: t_idx, f_idx, t_val_idx, f_val_idx
        # pero en tu caso sólo necesi tas t_idxs/f_idxs y times[f]/freqs[f]
        result = np.stack([t_idxs,
                           f_idxs,
                           np.round(times[t_idxs]*100).astype(int),   # ejemplo: tiempos*100 como int
                           np.round(freqs[f_idxs]).astype(int)], axis=1)
        # columnas: [t_idx, f_idx, t_cs, f_hz]
        return result

    def generate_hashes(self, peaks: np.ndarray):
        t_cs = peaks[:,2]
        f_hz = peaks[:,3]
        N    = len(peaks)
        for i in range(N):
            # fan-out dinámico: hasta fan_value vecinos
            for j in range(i+1, min(i + self.fan_value, N)):
                dt_cs = t_cs[j] - t_cs[i]
                df_hz = abs(f_hz[j] - f_hz[i])
                dt_s  = dt_cs / 100.0
                if 0 < dt_s <= self.zone_time and df_hz <= self.zone_freq:
                    # tripleta con resolución de 10 ms y ±Hz
                    h = f"{f_hz[i]}|{f_hz[j]}|{dt_cs}"
                    yield h, t_cs[i] / 100.0