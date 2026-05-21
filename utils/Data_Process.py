import numpy as np
import cupy as cp


def _bandpass_fir_taps(num_taps, low_cutoff, high_cutoff):
    n = np.arange(num_taps, dtype=np.float32) - np.float32((num_taps - 1) / 2)
    low = np.float32(low_cutoff / 2.0)
    high = np.float32(high_cutoff / 2.0)
    taps = 2 * high * np.sinc(2 * high * n) - 2 * low * np.sinc(2 * low * n)
    taps *= np.hamming(num_taps).astype(np.float32)
    return taps.astype(np.float32, copy=False)


def build_preprocess_context(Recon_Para, Para, paths):
    context = {
        "response_cache": {},
        "pinned_buffer": None,
        "pinned_shape": None,
        "pinned_dtype": None,
    }

    if Recon_Para.flag_sensitivity:
        sensitivity = np.load(paths.presettings_file("sensitivity_2D.npy"))
        context["sensitivity"] = np.asarray(sensitivity[:, :, Recon_Para.angles], dtype=np.float32)

    low_freq = 0.4 * Para.f0
    high_freq = 1.6 * Para.f0
    filter_order = 64
    nyquist = Para.fs / 2
    normalized_cutoff = [low_freq / nyquist, high_freq / nyquist]
    context["fir_taps"] = _bandpass_fir_taps(filter_order, normalized_cutoff[0], normalized_cutoff[1])
    context["fir_enabled"] = True
    return context


def _safe_fir_padlen(sample_count, fir_taps):
    default_padlen = 3 * (max(int(fir_taps.size), 1) - 1)
    return min(default_padlen, max(int(sample_count) - 1, 0))


def _next_power_of_two(value):
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _upload_to_torch(data, preprocess_context):
    import torch

    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if (
        preprocess_context.get("pinned_buffer") is None
        or preprocess_context.get("pinned_shape") != arr.shape
        or preprocess_context.get("pinned_dtype") != arr.dtype
    ):
        preprocess_context["pinned_buffer"] = torch.empty(arr.shape, dtype=torch.float32, pin_memory=True)
        preprocess_context["pinned_shape"] = arr.shape
        preprocess_context["pinned_dtype"] = arr.dtype

    pinned = preprocess_context["pinned_buffer"]
    pinned.copy_(torch.from_numpy(arr))
    return pinned.to(device="cuda", non_blocking=True)


def _analytic_response(preprocess_context, n_fft):
    import torch

    cache = preprocess_context["response_cache"]
    key = (int(n_fft), bool(preprocess_context.get("fir_enabled", True)))
    cached = cache.get(key)
    if cached is not None:
        return cached

    response = torch.zeros(int(n_fft), device="cuda", dtype=torch.float32)
    response[0] = 1.0
    if int(n_fft) % 2 == 0:
        response[1 : int(n_fft) // 2] = 2.0
        response[int(n_fft) // 2] = 1.0
    else:
        response[1 : (int(n_fft) + 1) // 2] = 2.0

    if preprocess_context.get("fir_enabled", True):
        fir_taps = torch.as_tensor(preprocess_context["fir_taps"], device="cuda", dtype=torch.float32)
        fir_freq = torch.fft.fft(fir_taps, n=int(n_fft))
        response *= torch.square(torch.abs(fir_freq)).to(torch.float32)

    response = response.reshape(int(n_fft), 1, 1)
    cache[key] = response
    return response


def _reflect_pad_samples(data_gpu, padlen):
    import torch

    if padlen <= 0:
        return data_gpu
    left = torch.flip(data_gpu[1 : padlen + 1], dims=(0,))
    right = torch.flip(data_gpu[-padlen - 1 : -1], dims=(0,))
    return torch.cat((left, data_gpu, right), dim=0)


def process_frame_data(data, Recon_Para, Para, preprocess_context):
    import torch
    from torch.utils import dlpack

    data_gpu = _upload_to_torch(data, preprocess_context)

    sensitivity = preprocess_context.get("sensitivity")
    if sensitivity is not None:
        if not isinstance(sensitivity, torch.Tensor):
            sensitivity = torch.as_tensor(sensitivity, device="cuda", dtype=torch.float32)
            preprocess_context["sensitivity"] = sensitivity
        data_gpu = data_gpu / sensitivity

    sample_count = int(data_gpu.shape[0])
    padlen = _safe_fir_padlen(sample_count, preprocess_context["fir_taps"])
    padded = _reflect_pad_samples(data_gpu, padlen)
    n_fft = _next_power_of_two(int(padded.shape[0]))
    response = _analytic_response(preprocess_context, n_fft)

    spectrum = torch.fft.fft(padded.to(torch.float32), n=n_fft, dim=0)
    analytic = torch.fft.ifft(spectrum * response, n=n_fft, dim=0)
    if padlen > 0:
        analytic = analytic[padlen : padlen + sample_count]
    analytic = analytic.to(torch.complex64).contiguous()
    analytic[0, :, :] = 0

    torch.cuda.synchronize()
    return cp.from_dlpack(dlpack.to_dlpack(analytic))
