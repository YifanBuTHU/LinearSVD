import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(data):
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return data.get()
    except ImportError:
        pass
    return np.asarray(data)


def format_db_image(data, dynamic_range):
    data_np = _to_numpy(data)
    magnitude = np.abs(data_np)
    max_val = np.max(magnitude)
    if max_val <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    magnitude = magnitude / max_val
    threshold = 10 ** (-dynamic_range / 20)
    magnitude[magnitude < threshold] = threshold
    return 20 * np.log10(magnitude)


def normalize_to_uint8(db_image):
    frame_min = db_image.min()
    frame_max = db_image.max()
    if frame_max <= frame_min:
        return np.zeros_like(db_image, dtype=np.uint8)
    return ((db_image - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)


def save_reconstruction_figure(recon_para, para, paths, data_recon=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_figure
    idx = recon_para.idx

    if data_recon is None:
        data_recon = np.load(paths.recon_frame_file(date, data_name, module, idx))

    nz, nx = data_recon.shape
    x_coords = np.arange(nx) * para.dx * 1000
    z_coords = np.arange(nz) * para.dz * 1000
    image_db = format_db_image(data_recon, dynamic_range)

    dpi = 200
    img_width_in = nx * 4 / dpi
    img_height_in = nz * 4 / dpi
    fig_width = img_width_in + 1.2
    fig_height = img_height_in + 0.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    im = ax.imshow(
        image_db,
        cmap="gray",
        extent=[x_coords[0], x_coords[-1], z_coords[-1], z_coords[0]],
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(f"{data_name}_{module}_{dynamic_range}dB_{idx}")

    if x_coords[-1] != x_coords[0]:
        ax.set_box_aspect((z_coords[-1] - z_coords[0]) / (x_coords[-1] - x_coords[0]))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Amplitude (dB)")

    plt.savefig(paths.figure_file(date, data_name, module, dynamic_range, idx), dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close()


def save_svd_figure(data_PDI, recon_para, paths, para=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    image_db = format_db_image(data_PDI, recon_para.dynamic_range_svd)

    nz, nx = data_PDI.shape
    if para is not None:
        x_coords = np.arange(nx) * para.dx * 1000
        z_coords = np.arange(nz) * para.dz * 1000
        extent = [x_coords[0], x_coords[-1], z_coords[-1], z_coords[0]]
        xlabel = "X (mm)"
        ylabel = "Z (mm)"
        aspect = "equal"
        box_aspect = (z_coords[-1] - z_coords[0]) / (x_coords[-1] - x_coords[0]) if (x_coords[-1] != x_coords[0]) else 1
    else:
        extent = [0, nx, nz, 0]
        xlabel = "X (pixel)"
        ylabel = "Z (pixel)"
        aspect = "equal"
        box_aspect = nz / nx if nx != 0 else 1

    dpi = 200
    img_width_in = nx * 4 / dpi
    img_height_in = nz * 4 / dpi
    fig_width = img_width_in + 1.2
    fig_height = img_height_in + 0.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    im = ax.imshow(image_db, cmap="hot", extent=extent, aspect=aspect, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{recon_para.Data_Name}_{recon_para.Module}_SVD{recon_para.SVD_num}_{recon_para.dynamic_range_svd}dB"
    )
    if nx > 0 and nz > 0:
        ax.set_box_aspect(box_aspect)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Amplitude (dB)")

    plt.savefig(
        paths.svd_figure_file(
            recon_para.Date,
            recon_para.Data_Name,
            recon_para.Module,
            recon_para.SVD_num,
            recon_para.dynamic_range_svd,
        ),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close()


def create_video_from_stack(recon_para, data, paths, progress_callback=None, logger=None):
    logger = logger or print
    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_video

    output_dir = paths.video_dir(date)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = str(paths.video_file(date, data_name, module))

    data_np = _to_numpy(data)
    _, _, frame_num = data_np.shape
    first_frame_processed = format_db_image(data_np[:, :, 0], dynamic_range)
    processed_height, processed_width = first_frame_processed.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_name, fourcc, 30, (processed_width, processed_height))

    logger(f"Generating video from data with {frame_num} frames...")
    if progress_callback:
        progress_callback({"stage": "video", "current": 0, "total": frame_num, "message": "Generating video"})

    for idx in range(frame_num):
        try:
            current_frame = data_np[:, :, idx]
            frame_processed = format_db_image(current_frame, dynamic_range)
            frame_normalized = normalize_to_uint8(frame_processed)
            frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame_bgr)
            if progress_callback:
                progress_callback(
                    {"stage": "video", "current": idx + 1, "total": frame_num, "message": "Generating video"}
                )
        except Exception as exc:
            logger(f"Error processing frame {idx}: {exc}")
            continue

    video_writer.release()
    logger(f"Video saved as: {video_name}")


def create_video_from_files(recon_para, paths, progress_callback=None, logger=None):
    logger = logger or print
    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_video
    frame_num = recon_para.frame_num

    output_dir = paths.video_dir(date)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = str(paths.video_file(date, data_name, module))

    first_frame = np.load(paths.recon_frame_file(date, data_name, module, 0))
    first_frame_processed = format_db_image(first_frame, dynamic_range)
    height, width = first_frame_processed.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    logger(f"Processing {frame_num} frames for video generation...")
    if progress_callback:
        progress_callback({"stage": "video", "current": 0, "total": frame_num, "message": "Generating video"})

    for idx in range(frame_num):
        try:
            data_recon = np.load(paths.recon_frame_file(date, data_name, module, idx))
            frame_processed = format_db_image(data_recon, dynamic_range)
            frame_normalized = normalize_to_uint8(frame_processed)
            frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame_bgr)
            if progress_callback:
                progress_callback(
                    {"stage": "video", "current": idx + 1, "total": frame_num, "message": "Generating video"}
                )
        except FileNotFoundError:
            logger(f"Warning: file for frame {idx} not found, skipping...")
            continue
        except Exception as exc:
            logger(f"Error processing frame {idx}: {exc}")
            continue

    video_writer.release()
    logger(f"Video saved as: {video_name}")
