import cupy as cp
import cupyx.scipy.ndimage as cndimage
import numpy as np
import time

from visualization import create_video_from_stack, save_svd_figure


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

    def _format_index_list(self, indices):
        indices = list(indices)
        if not indices:
            return "None"
        if len(indices) <= 50:
            return ", ".join(str(idx) for idx in indices)
        preview = ", ".join(str(idx) for idx in indices[:20])
        return f"{preview}, ... ({len(indices)} total, last={indices[-1]})"

    def SVD(self, frame_indices=None, progress_callback=None, logger=None):
        log = logger or print

        log("Loading reconstruction stack for SVD...")
        self._emit_progress(progress_callback, "svd_load", 0, 1, "Loading SVD data")
        stage_start = time.perf_counter()
        data, used_indices = self.load_files(
            f"Recon_{self.Data_Name}_{self.Module}",
            self.directory,
            frame_indices=frame_indices,
        )
        self._emit_progress(progress_callback, "svd_load", 1, 1, "SVD data loaded")
        log(f"SVD data loaded in {time.perf_counter() - stage_start:.3f}s")

        log(f"SVD will use {len(used_indices)} reconstructed frames: {self._format_index_list(used_indices)}")

        log("Computing SVD...")
        self._emit_progress(progress_callback, "svd_compute", 0, 1, "Computing SVD")
        stage_start = time.perf_counter()
        needs_svd_stack = self.Recon_Para.flag_SVD_Frames or self.Recon_Para.flag_SVD_Video
        if needs_svd_stack:
            data_svd, data_svd_noise = self.SVD_Cal(data, self.SVD_num)
        else:
            data_svd = None
            data_svd_noise = None
            data_PDI, data_PDI_noise = self.SVD_PDI_Cal(data, self.SVD_num)
        self._emit_progress(progress_callback, "svd_compute", 1, 1, "SVD computed")
        log(f"SVD compute completed in {time.perf_counter() - stage_start:.3f}s")

        if self.Recon_Para.flag_SVD_Frames:
            log("Saving SVD filtered complex frames...")
            stage_start = time.perf_counter()
            self.save_svd_frames(data_svd, used_indices, progress_callback=progress_callback)
            log(f"SVD filtered frames saved in {time.perf_counter() - stage_start:.3f}s")

        if self.Recon_Para.flag_SVD_Video:
            log("Generating SVD filtered video...")
            stage_start = time.perf_counter()
            create_video_from_stack(
                self.Recon_Para,
                data_svd,
                self.paths,
                progress_callback=progress_callback,
                logger=log,
                video_path=self.paths.svd_video_file(self.Date, self.Data_Name, self.Module, self.SVD_num),
                stage="svd_video",
                message="Generating SVD video",
            )
            log(f"SVD filtered video generated in {time.perf_counter() - stage_start:.3f}s")

        log("Estimating noise profile...")
        self._emit_progress(progress_callback, "svd_noise", 0, 1, "Estimating noise")
        stage_start = time.perf_counter()
        if needs_svd_stack:
            data_PDI = self.pdi_from_svd_stack(data_svd)
            data_PDI_noise = self.pdi_from_svd_stack(data_svd_noise)
        data_PDI_noise = self.Noise_Estimation(data_PDI_noise)
        data_PDI = data_PDI / data_PDI_noise
        self._emit_progress(progress_callback, "svd_noise", 1, 1, "Noise estimation complete")
        log(f"SVD noise estimation completed in {time.perf_counter() - stage_start:.3f}s")

        log("Saving SVD figure...")
        self._emit_progress(progress_callback, "svd_figure", 0, 1, "Saving SVD figure")
        stage_start = time.perf_counter()
        save_svd_figure(data_PDI, self.Recon_Para, self.paths, self.para)
        self._emit_progress(progress_callback, "svd_figure", 1, 1, "SVD figure saved")
        log(f"SVD figure saved in {time.perf_counter() - stage_start:.3f}s")
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

    def save_svd_frames(self, data_svd, used_indices, progress_callback=None):
        output_dir = self.paths.svd_frame_dir(self.Date)
        output_dir.mkdir(parents=True, exist_ok=True)
        total = len(used_indices)
        self._emit_progress(progress_callback, "svd_frames", 0, total, "Saving SVD frames")

        for stack_idx, frame_idx in enumerate(used_indices):
            output_path = self.paths.svd_frame_file(self.Date, self.Data_Name, self.Module, self.SVD_num, frame_idx)
            np.save(output_path, cp.asnumpy(data_svd[:, :, stack_idx]))
            self._emit_progress(
                progress_callback,
                "svd_frames",
                stack_idx + 1,
                total,
                "Saving SVD frames",
            )

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

        data_SVD = cp.dot(U, S[:, None] * Vh)
        data_SVD_noise = cp.dot(U, S_noise[:, None] * Vh)

        del U, S, Vh, S_noise
        cp.get_default_memory_pool().free_all_blocks()

        return data_SVD.reshape(data.shape), data_SVD_noise.reshape(data.shape)

    def SVD_PDI_Cal(self, data, SVD_num):
        frame_count = data.shape[-1]
        data_reshaped = data.reshape(-1, frame_count)
        U, S, Vh = cp.linalg.svd(data_reshaped, full_matrices=False)

        component_count = int(S.shape[0])
        svd_start = min(max(int(SVD_num), 0), component_count)
        noise_start = max(svd_start, max(component_count - 4, 0))

        data_PDI = self.pdi_from_svd_components(U, S, data.shape, frame_count, slice(svd_start, component_count))
        data_PDI_noise = self.pdi_from_svd_components(U, S, data.shape, frame_count, slice(noise_start, component_count))

        del U, S, Vh
        cp.get_default_memory_pool().free_all_blocks()
        return data_PDI, data_PDI_noise

    def pdi_from_svd_components(self, U, singular_values, data_shape, frame_count, component_slice):
        components = U[:, component_slice]
        values = singular_values[component_slice]
        if int(values.shape[0]) == 0:
            return cp.zeros(data_shape[:2], dtype=cp.float32)

        power = cp.sum((cp.abs(components) ** 2) * (values[None, :] ** 2), axis=1) / frame_count
        data_PDI = power.reshape(data_shape[:2])
        max_value = cp.max(data_PDI)
        if max_value > 0:
            data_PDI = data_PDI / max_value
        return data_PDI

    def pdi_from_svd_stack(self, data_svd):
        data_PDI = cp.abs(data_svd)
        data_PDI = data_PDI ** 2
        data_PDI = cp.sum(data_PDI, axis=-1) / data_svd.shape[-1]
        max_value = cp.max(data_PDI)
        if max_value > 0:
            data_PDI = data_PDI / max_value
        return data_PDI

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
