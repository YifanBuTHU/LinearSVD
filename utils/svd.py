import cupy as cp
import cupyx.scipy.ndimage as cndimage
import numpy as np

from visualization import save_svd_figure


class SVD_filter:
    def __init__(self, Recon_Para, paths, para=None):
        self.frame_num = Recon_Para.frame_num
        self.Data_Name = Recon_Para.Data_Name
        self.Module = Recon_Para.Module
        self.Date = Recon_Para.Date
        self.dynamic_range_svd = Recon_Para.dynamic_range_svd
        self.idx = Recon_Para.idx
        self.SVD_num = Recon_Para.SVD_num
        self.directory = paths.session_data_dir(self.Date)
        self.paths = paths
        self.Recon_Para = Recon_Para
        self.para = para

    def _emit_progress(self, progress_callback, stage, current, total, message):
        if progress_callback is None:
            return
        progress_callback(
            {
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
            }
        )

    def SVD(self, frame_indices=None, progress_callback=None, logger=None):
        log = logger or print

        log("Loading reconstruction stack for SVD...")
        self._emit_progress(progress_callback, "svd_load", 0, 1, "Loading SVD data")
        data, used_indices = self.load_files(
            f"Recon_{self.Data_Name}_{self.Module}",
            self.directory,
            frame_indices=frame_indices,
        )
        self._emit_progress(progress_callback, "svd_load", 1, 1, "SVD data loaded")

        log(f"SVD will use {len(used_indices)} reconstructed frames: {used_indices}")

        log("Computing SVD...")
        self._emit_progress(progress_callback, "svd_compute", 0, 1, "Computing SVD")
        data_PDI, data_PDI_noise = self.SVD_Cal(data, self.SVD_num)
        self._emit_progress(progress_callback, "svd_compute", 1, 1, "SVD computed")

        log("Estimating noise profile...")
        self._emit_progress(progress_callback, "svd_noise", 0, 1, "Estimating noise")
        data_PDI_noise = self.Noise_Estimation(data_PDI_noise)
        data_PDI = data_PDI / data_PDI_noise
        self._emit_progress(progress_callback, "svd_noise", 1, 1, "Noise estimation complete")

        log("Saving SVD figure...")
        self._emit_progress(progress_callback, "svd_figure", 0, 1, "Saving SVD figure")
        save_svd_figure(data_PDI, self.Recon_Para, self.paths, self.para)
        self._emit_progress(progress_callback, "svd_figure", 1, 1, "SVD figure saved")
        return data_PDI, used_indices

    def load_files(self, data_name, directory, frame_indices=None):
        files = [path.name for path in directory.iterdir() if path.is_file()]
        npy_files = [name for name in files if name.startswith(data_name) and name.endswith(".npy")]
        npy_files.sort(key=lambda name: int(name.split("_")[-1].split(".")[0]))

        if frame_indices is not None:
            allowed = set(int(idx) for idx in frame_indices)
            npy_files = [name for name in npy_files if int(name.split("_")[-1].split(".")[0]) in allowed]

        if not npy_files:
            raise FileNotFoundError(f"No reconstruction files found in {directory} matching {data_name}_*.npy")

        used_indices = [int(name.split("_")[-1].split(".")[0]) for name in npy_files]
        data_np = np.stack([np.load(directory / name) for name in npy_files], axis=-1)
        return cp.asarray(data_np), used_indices

    def Registration(self, data):
        interpolation_factor = 5
        _, _, frame_count = data.shape
        registered = cp.empty_like(data)

        ref_frame_complex = data[:, :, 0]
        registered[:, :, 0] = ref_frame_complex

        ref_mag = cp.abs(ref_frame_complex).astype(cp.float32)
        ref_mag_large = cndimage.zoom(ref_mag, interpolation_factor, order=3)
        fft_base = cp.fft.fft2(ref_mag_large)

        for idx in range(1, frame_count):
            frame_complex = data[:, :, idx]
            cur_mag = cp.abs(frame_complex).astype(cp.float32)
            cur_mag_large = cndimage.zoom(cur_mag, interpolation_factor, order=3)
            fft_current = cp.fft.fft2(cur_mag_large)

            correlation = fft_base * cp.conj(fft_current)
            correlation /= cp.abs(correlation) + 1e-8
            corr = cp.fft.ifft2(correlation)
            corr = cp.fft.fftshift(corr)
            corr_abs = cp.abs(corr)
            corr_abs = corr_abs / cp.max(corr_abs)

            maxloc = cp.unravel_index(cp.argmax(corr_abs), corr_abs.shape)
            center = cp.array(corr_abs.shape) // 2
            dy_large = maxloc[0] - center[0]
            dx_large = maxloc[1] - center[1]

            dy = dy_large / interpolation_factor
            dx = dx_large / interpolation_factor
            registered[:, :, idx] = self._apply_shift_subpixel(frame_complex, dy, dx)

        return registered

    def _apply_shift_subpixel(self, img, dy, dx):
        real_shift = cndimage.shift(img.real, shift=(-dy, -dx), order=3, mode="constant", cval=0)
        imag_shift = cndimage.shift(img.imag, shift=(-dy, -dx), order=3, mode="constant", cval=0)
        return real_shift + 1j * imag_shift

    def SVD_Cal(self, data, SVD_num):
        data_reshaped = data.reshape(-1, data.shape[-1])
        U, S, Vh = cp.linalg.svd(data_reshaped, full_matrices=False)

        S[0:SVD_num] = 0
        S_noise = S.copy()
        S_noise[0:-4] = 0

        data_SVD = cp.dot(U, cp.dot(cp.diag(S), Vh))
        data_SVD_noise = cp.dot(U, cp.dot(cp.diag(S_noise), Vh))

        del U, S, Vh, S_noise
        cp.get_default_memory_pool().free_all_blocks()

        data_recon = data_SVD.reshape(data.shape)
        data_recon_noise = data_SVD_noise.reshape(data.shape)

        data_PDI = cp.abs(data_recon)
        data_PDI = data_PDI ** 2
        data_PDI = cp.sum(data_PDI, axis=-1) / data.shape[-1]
        data_PDI = data_PDI / cp.max(data_PDI)

        data_PDI_noise = cp.abs(data_recon_noise)
        data_PDI_noise = data_PDI_noise ** 2
        data_PDI_noise = cp.sum(data_PDI_noise, axis=-1) / data.shape[-1]
        data_PDI_noise = data_PDI_noise / cp.max(data_PDI_noise)
        return data_PDI, data_PDI_noise

    def Noise_Estimation(self, data_PDI_noise):
        profile = cp.mean(data_PDI_noise, axis=1)
        window_length = max(1, data_PDI_noise.shape[1] // 6)
        weights = cp.ones(window_length, dtype=profile.dtype)
        num = cp.convolve(profile, weights, mode="same")
        denom = cp.convolve(cp.ones_like(profile), weights, mode="same")
        smoothed_profile = num / denom
        noise_profile = smoothed_profile.reshape(-1, 1)
        noise_profile = cp.repeat(noise_profile, repeats=data_PDI_noise.shape[1], axis=1)

        profile = cp.mean(data_PDI_noise, axis=0)
        window_length = max(1, data_PDI_noise.shape[0] // 6)
        weights = cp.ones(window_length, dtype=profile.dtype)
        num = cp.convolve(profile, weights, mode="same")
        denom = cp.convolve(cp.ones_like(profile), weights, mode="same")
        smoothed_profile = num / denom
        return noise_profile * smoothed_profile
