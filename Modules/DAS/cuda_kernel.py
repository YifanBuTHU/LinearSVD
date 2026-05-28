import os
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "CUPY_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".cupy_kernel_cache"),
)

import cupy as cp


_DAS_FUSED_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void das_fused_kernel(
    const float2* __restrict__ data,
    const int* __restrict__ sample_rf,
    const float* __restrict__ weights,
    float2* __restrict__ output,
    const int segment_nx,
    const int nz,
    const int num_ele,
    const int num_samples,
    const int angle_count
) {
    const int pixel = blockIdx.x;
    const int rx = threadIdx.x;
    const int total_pixels = segment_nx * nz;
    if (pixel >= total_pixels) {
        return;
    }

    const int z = pixel % nz;
    const int x = pixel / nz;
    const int pixel_ele_base = (x * nz + z) * num_ele;
    const float inv_norm = 1.0f / ((float)num_ele * (float)angle_count);

    __shared__ float partial_real[256];
    __shared__ float partial_imag[256];

    float local_real = 0.0f;
    float local_imag = 0.0f;

    if (rx < num_ele) {
        const float weight = weights[pixel_ele_base + rx];
        const int sample_base = (pixel_ele_base + rx) * angle_count;

        for (int angle = 0; angle < angle_count; ++angle) {
            const int sample_idx = sample_rf[sample_base + angle];
            if (sample_idx > 0 && sample_idx < num_samples) {
                const float2 sample = data[(sample_idx * num_ele + rx) * angle_count + angle];
                local_real += sample.x * weight;
                local_imag += sample.y * weight;
            }
        }
    }

    partial_real[rx] = local_real;
    partial_imag[rx] = local_imag;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (rx < stride) {
            partial_real[rx] += partial_real[rx + stride];
            partial_imag[rx] += partial_imag[rx + stride];
        }
        __syncthreads();
    }

    if (rx == 0) {
        output[pixel] = make_float2(partial_real[0] * inv_norm, partial_imag[0] * inv_norm);
    }
}
""",
    "das_fused_kernel",
    options=("--std=c++11",),
)


def reconstruct_segment(data, sample_rf, weights, num_ele, angle_count):
    if int(num_ele) > 256:
        raise ValueError(f"DAS CUDA kernel supports up to 256 channels, got {num_ele}.")

    sample_rf_c = cp.ascontiguousarray(sample_rf.astype(cp.int32, copy=False))
    weights_c = cp.ascontiguousarray(weights.astype(cp.float32, copy=False))
    segment_nx, nz = int(sample_rf_c.shape[0]), int(sample_rf_c.shape[1])
    output = cp.empty((segment_nx, nz), dtype=cp.complex64)

    _DAS_FUSED_KERNEL(
        (segment_nx * nz,),
        (256,),
        (
            data,
            sample_rf_c,
            weights_c,
            output,
            np.int32(segment_nx),
            np.int32(nz),
            np.int32(num_ele),
            np.int32(data.shape[0]),
            np.int32(angle_count),
        ),
    )
    return output


def das_fused_reconstruction(data, segments, nx, nz, num_ele, angle_count):
    data_c = cp.ascontiguousarray(data.astype(cp.complex64, copy=False))
    data_c[0, :, :] = 0
    data_recon = cp.empty((int(nx), int(nz)), dtype=cp.complex64)

    for segment in segments:
        data_recon[segment["x_slice"], :] = reconstruct_segment(
            data_c,
            segment["sample_rf"],
            segment["weights"],
            num_ele,
            angle_count,
        )

    return data_recon
