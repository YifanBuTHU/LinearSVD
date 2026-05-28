import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cupy as cp
import numpy as np
from tqdm import tqdm

from pipeline.progress import PipelineProgressReporter
from utils.Data_Process import build_preprocess_context, process_frame_data
from utils.Dataset import TsingPaiDataset
from utils.Linear_TOF_Cal import Linear_TOF_Cal
from utils.config import PathConfig, ReconParams
from utils.svd import SVD_filter
from visualization import create_video_from_files, save_reconstruction_figure


PER_FRAME_METRICS = (
    "load_s",
    "preprocess_s",
    "wait_preprocess_s",
    "upload_s",
    "das_s",
    "save_submit_s",
    "save_s",
    "frame_wall_s",
    "total_stage_s",
)


def load_reconstruction_callable(recon_para):
    module = importlib.import_module(f"Modules.{recon_para.Module}.Reconstruction")
    return module.Reconstruction


def main_recon(reconstruction, cp_data, recon_context, para, recon_para):
    data_recon_temp = reconstruction(cp_data, recon_context, para, recon_para)
    data_recon = cp.asnumpy(data_recon_temp) if isinstance(data_recon_temp, cp.ndarray) else np.asarray(data_recon_temp)
    return np.flipud(data_recon.T)


def _format_index_list(indices):
    indices = list(indices)
    if not indices:
        return "None"
    if len(indices) <= 50:
        return ", ".join(str(idx) for idx in indices)
    preview = ", ".join(str(idx) for idx in indices[:20])
    return f"{preview}, ... ({len(indices)} total, last={indices[-1]})"


def _build_frame_records(recon_para, paths, dataset=None):
    expected_emissions = int(recon_para.angles.size)
    records = []
    dataset_length = len(dataset) if dataset is not None else None

    for idx in range(recon_para.frame_num):
        record = {
            "idx": idx,
            "frame_name": None,
            "raw_emissions": None,
            "issues": [],
            "included_preprocess": False,
            "included_recon": False,
            "included_svd": False,
            "metrics": {},
        }

        if dataset is not None:
            if idx >= dataset_length:
                record["issues"].append(f"raw frame missing (requested index {idx}, available frames: {dataset_length})")
            else:
                record["frame_name"] = dataset.frame_name(idx)
                record["raw_emissions"] = dataset.frame_emission_count(idx)
                if record["raw_emissions"] != expected_emissions:
                    record["issues"].append(
                        f"raw frame has {record['raw_emissions']} emissions, expected {expected_emissions}"
                    )

        records.append(record)

    return records


def _percentile(values, fraction):
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    index = int(np.ceil(float(fraction) * len(sorted_values))) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def _metric_values(records, key):
    values = []
    for record in records:
        value = record.get("metrics", {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def _build_performance_lines(records, run_metrics):
    lines = [
        "Performance metrics:",
        f"Pipeline wall time: {run_metrics.get('wall_s', 0.0):.3f} s",
        f"TOF total: {run_metrics.get('tof_s', 0.0):.3f} s",
        f"Preprocess workers: {run_metrics.get('preprocess_workers', 0)}",
        f"Async save: {run_metrics.get('async_save', False)}",
        f"Reconstructed frames: {sum(1 for r in records if r['included_recon'])}",
    ]

    for key in PER_FRAME_METRICS:
        values = _metric_values(records, key)
        if not values:
            continue
        lines.append(
            f"- {key}: mean={np.mean(values):.4f}s, p95={_percentile(values, 0.95):.4f}s, "
            f"total={np.sum(values):.4f}s, n={len(values)}"
        )
    return lines


def _write_run_summary(paths, recon_para, records, svd_status, run_metrics=None):
    run_metrics = run_metrics or {}
    summary_path = paths.result_date_dir(recon_para.Date) / f"{recon_para.Data_Name}_{recon_para.Module}_run_summary.txt"
    included_pre = [r["idx"] for r in records if r["included_preprocess"]]
    included_recon = [r["idx"] for r in records if r["included_recon"]]
    included_svd = [r["idx"] for r in records if r["included_svd"]]
    problem_records = [r for r in records if r["issues"]]

    lines = [
        f"Data name: {recon_para.Data_Name}",
        f"Date: {recon_para.Date}",
        f"Module: {recon_para.Module}",
        f"Requested frame count: {recon_para.frame_num}",
        f"Configured angles: {recon_para.angles.tolist()}",
        f"Preprocessed frame indices: {_format_index_list(included_pre)}",
        f"Reconstructed frame indices: {_format_index_list(included_recon)}",
        f"SVD included frame indices: {_format_index_list(included_svd)}",
        f"SVD status: {svd_status}",
        "",
    ]

    if recon_para.emit_perf_metrics:
        lines.extend(_build_performance_lines(records, run_metrics))
        lines.append("")

    lines.append("Problem frames:")
    if not problem_records:
        lines.append("None")
    else:
        for record in problem_records:
            name = record["frame_name"] or "<no raw frame>"
            issues = "; ".join(record["issues"])
            lines.append(
                f"- idx={record['idx']}, name={name}, raw_emissions={record['raw_emissions']}, "
                f"included_preprocess={record['included_preprocess']}, included_recon={record['included_recon']}, "
                f"included_svd={record['included_svd']}, issues={issues}"
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _log_problem_summary(log, records, summary_path):
    problem_records = [r for r in records if r["issues"]]
    if not problem_records:
        log("No problematic frames were detected.")
        log(f"Run summary written to: {summary_path}")
        return

    log(f"Detected {len(problem_records)} problematic frames. They were excluded from reconstruction/SVD.")
    for record in problem_records:
        name = record["frame_name"] or "<no raw frame>"
        log(f"  Frame idx={record['idx']} ({name}): {'; '.join(record['issues'])}")
    log(f"Run summary written to: {summary_path}")


def _collect_recon_indices_for_svd(recon_para, paths, records):
    indices = []
    for record in records:
        idx = record["idx"]
        recon_file = paths.recon_frame_file(recon_para.Date, recon_para.Data_Name, recon_para.Module, idx)
        if record["issues"]:
            continue
        if recon_file.exists():
            indices.append(idx)
    return indices


def _resolve_preprocess_workers(recon_para, frame_count):
    if frame_count <= 0:
        return 1
    if recon_para.preprocess_workers is not None:
        return max(1, min(int(recon_para.preprocess_workers), int(frame_count)))
    # Long mouse runs showed SciPy's filtfilt/hilbert path is more stable with
    # one worker; callers can still opt into a higher hidden value explicitly.
    return 1


def _load_and_preprocess_frame(dataset, idx, recon_para, para, preprocess_context):
    start = time.perf_counter()
    raw_data = dataset[idx]
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    cp_data = process_frame_data(raw_data, recon_para, para, preprocess_context)
    preprocess_s = time.perf_counter() - start

    return {
        "idx": idx,
        "cp_data": cp_data,
        "load_s": load_s,
        "preprocess_s": preprocess_s,
    }


def _save_frame_outputs(paths, recon_para, para, idx, data_recon):
    start = time.perf_counter()
    np.save(paths.recon_frame_file(recon_para.Date, recon_para.Data_Name, recon_para.Module, idx), data_recon)
    if recon_para.flag_Figure:
        frame_recon_para = replace(recon_para, idx=idx)
        save_reconstruction_figure(frame_recon_para, para, paths, data_recon=data_recon)
    return time.perf_counter() - start


def _warm_reconstruction_kernels(reconstruction, recon_context, para, recon_para):
    angle_count = int(recon_context["angle_count"])
    warm_data = cp.zeros((int(para.num_samples), int(para.num_ele), angle_count), dtype=cp.complex64)
    _ = reconstruction(warm_data, recon_context, para, recon_para)
    cp.cuda.Stream.null.synchronize()


def _collect_save_results(pending_saves, records):
    for idx, future in pending_saves:
        save_s = future.result()
        records[idx]["metrics"]["save_s"] = save_s
        metrics = records[idx]["metrics"]
        metrics["total_stage_s"] = (
            metrics.get("load_s", 0.0)
            + metrics.get("preprocess_s", 0.0)
            + metrics.get("upload_s", 0.0)
            + metrics.get("das_s", 0.0)
            + save_s
        )


def _run_streaming_reconstruction(
    valid_indices,
    records,
    recon_para,
    para,
    paths,
    dataset,
    preprocess_context,
    reconstruction,
    recon_context,
    use_tqdm,
    log,
    progress_reporter,
):
    total = len(valid_indices)
    if total == 0:
        log("No valid frames are available for reconstruction.")
        return {"preprocess_workers": 0, "async_save": bool(recon_para.async_save)}

    preprocess_workers = _resolve_preprocess_workers(recon_para, total)
    log(
        "Streaming raw preprocessing and reconstruction "
        f"with {preprocess_workers} preprocess worker(s)."
    )
    log("Processed frame cache is disabled; raw frames stream directly into reconstruction.")

    progress_reporter.emit("stream_pipeline", 0, total, message="Processing and reconstructing frames")
    iterator = tqdm(valid_indices, desc="Processing and reconstructing frames", unit="frame") if use_tqdm else valid_indices
    save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="linear-svd-save") if recon_para.async_save else None
    pending_saves = []

    try:
        with ThreadPoolExecutor(max_workers=preprocess_workers, thread_name_prefix="linear-svd-preprocess") as preprocess_executor:
            pending_preprocess = {}
            next_submit = 0

            def submit_next():
                nonlocal next_submit
                if next_submit >= total:
                    return
                frame_idx = valid_indices[next_submit]
                pending_preprocess[frame_idx] = preprocess_executor.submit(
                    _load_and_preprocess_frame,
                    dataset,
                    frame_idx,
                    recon_para,
                    para,
                    preprocess_context,
                )
                next_submit += 1

            for _ in range(min(preprocess_workers, total)):
                submit_next()

            for position, idx in enumerate(iterator, start=1):
                frame_wall_start = time.perf_counter()
                recon_para.idx = idx

                wait_start = time.perf_counter()
                preprocessed = pending_preprocess.pop(idx).result()
                wait_preprocess_s = time.perf_counter() - wait_start

                upload_start = time.perf_counter()
                cp_data = preprocessed["cp_data"]
                upload_s = time.perf_counter() - upload_start

                das_start = time.perf_counter()
                data_recon = main_recon(reconstruction, cp_data, recon_context, para, recon_para)
                das_s = time.perf_counter() - das_start

                metrics = records[idx]["metrics"]
                metrics.update(
                    {
                        "load_s": preprocessed["load_s"],
                        "preprocess_s": preprocessed["preprocess_s"],
                        "wait_preprocess_s": wait_preprocess_s,
                        "upload_s": upload_s,
                        "das_s": das_s,
                    }
                )
                records[idx]["included_preprocess"] = True
                records[idx]["included_recon"] = True

                save_submit_start = time.perf_counter()
                if save_executor is not None:
                    pending_saves.append(
                        (
                            idx,
                            save_executor.submit(
                                _save_frame_outputs,
                                paths,
                                recon_para,
                                para,
                                idx,
                                data_recon,
                            ),
                        )
                    )
                else:
                    save_s = _save_frame_outputs(paths, recon_para, para, idx, data_recon)
                    metrics["save_s"] = save_s
                metrics["save_submit_s"] = time.perf_counter() - save_submit_start
                metrics["frame_wall_s"] = time.perf_counter() - frame_wall_start
                metrics["total_stage_s"] = (
                    metrics.get("load_s", 0.0)
                    + metrics.get("preprocess_s", 0.0)
                    + metrics.get("upload_s", 0.0)
                    + metrics.get("das_s", 0.0)
                    + metrics.get("save_s", 0.0)
                )

                submit_next()
                del cp_data, data_recon, preprocessed

                progress_reporter.emit(
                    "stream_pipeline",
                    position,
                    total,
                    message=f"Processing and reconstructing frames ({position}/{total})",
                )
    finally:
        _collect_save_results(pending_saves, records)
        if save_executor is not None:
            save_executor.shutdown(wait=True)

    log("Processing and reconstructing frames finished.")
    return {"preprocess_workers": preprocess_workers, "async_save": bool(recon_para.async_save)}


def run_pipeline(recon_para, para, paths, use_tqdm=True, logger=None, progress_callback=None):
    log = logger or print
    start_time = time.perf_counter()
    progress_reporter = PipelineProgressReporter(
        recon_para,
        progress_callback=progress_callback,
        start_time=start_time,
    )
    run_metrics = {"tof_s": 0.0, "preprocess_workers": 0, "async_save": bool(recon_para.async_save)}

    if recon_para.gpu_device is not None:
        cp.cuda.Device(recon_para.gpu_device).use()
        log(f"Using GPU device: {recon_para.gpu_device}")

    recon_para.apply_to_probe(para)
    recon_para.validate(para, paths)
    paths.ensure_runtime_dirs(recon_para)

    log(f"Data root: {paths.data_root}")
    log(f"Results root: {paths.results_root}")
    log(f"Session date: {recon_para.Date}")
    log(f"Data name: {recon_para.Data_Name}")

    dataset = None
    raw_dir = paths.unpacked_dir(recon_para.Date, recon_para.Data_Name)
    if raw_dir.exists():
        dataset = TsingPaiDataset(raw_dir, para, verbose=False)
        log(f"Indexed raw dataset from: {raw_dir}")

    records = _build_frame_records(recon_para, paths, dataset=dataset)
    valid_indices = [record["idx"] for record in records if not record["issues"]]

    skipped_count = len(records) - len(valid_indices)
    if skipped_count:
        log(f"Skipping {skipped_count} problematic frames before computation.")

    if recon_para.flag_Recon:
        if dataset is None:
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        preprocess_context = build_preprocess_context(recon_para, para, paths)
        log("Preparing reconstruction tables and apodization weights...")
        progress_reporter.emit("recon_prepare", 0, 1, message="Preparing reconstruction")
        reconstruction = load_reconstruction_callable(recon_para)
        tof_start = time.perf_counter()
        recon_context = Linear_TOF_Cal(para, recon_para)
        cp.cuda.Stream.null.synchronize()
        run_metrics["tof_s"] = time.perf_counter() - tof_start
        progress_reporter.emit("recon_prepare", 1, 1, message="Reconstruction preparation complete")
        log(f"TOF calculation completed in {run_metrics['tof_s']:.3f}s")
        _warm_reconstruction_kernels(reconstruction, recon_context, para, recon_para)

        stream_metrics = _run_streaming_reconstruction(
            valid_indices,
            records,
            recon_para,
            para,
            paths,
            dataset,
            preprocess_context,
            reconstruction,
            recon_context,
            use_tqdm,
            log,
            progress_reporter,
        )
        run_metrics.update(stream_metrics)

    if recon_para.flag_Video:
        log("Generating reconstruction video from reconstructed frames...")
        create_video_from_files(
            recon_para,
            paths,
            progress_callback=progress_reporter.callback("video", "Generating reconstruction video"),
            logger=log,
        )

    svd_status = "not requested"
    if recon_para.flag_SVD:
        svd_indices = _collect_recon_indices_for_svd(recon_para, paths, records)
        if len(svd_indices) <= recon_para.SVD_num:
            svd_status = (
                f"skipped: need more reconstructed frames than SVD_num ({len(svd_indices)} available, "
                f"SVD_num={recon_para.SVD_num})"
            )
            log(f"Skipping SVD: {svd_status}")
        elif len(svd_indices) < 2:
            svd_status = f"skipped: need at least 2 reconstructed frames for SVD, got {len(svd_indices)}"
            log(f"Skipping SVD: {svd_status}")
        else:
            log("Running SVD filtering...")
            svd_filter = SVD_filter(recon_para, paths, para)
            _, used_indices = svd_filter.SVD(
                frame_indices=svd_indices,
                progress_callback=progress_reporter.callback("svd_compute", "Running SVD"),
                logger=log,
            )
            for idx in used_indices:
                if 0 <= idx < len(records):
                    records[idx]["included_svd"] = True
            svd_status = f"completed using frame indices: {_format_index_list(used_indices)}"

    run_metrics["wall_s"] = time.perf_counter() - start_time
    summary_path = _write_run_summary(paths, recon_para, records, svd_status, run_metrics=run_metrics)
    _log_problem_summary(log, records, summary_path)

    log(f"Run completed. Results available at: {paths.result_date_dir(recon_para.Date)}")
    progress_reporter.emit("completed", 1, 1, message="Run completed")
    return {
        "results_dir": paths.result_date_dir(recon_para.Date),
        "session_data_dir": paths.session_data_dir(recon_para.Date),
        "summary_file": summary_path,
        "performance": run_metrics,
    }


def main(recon_para, para, paths, use_tqdm=True, logger=None):
    return run_pipeline(recon_para, para, paths, use_tqdm=use_tqdm, logger=logger)


def example_main():
    from utils.Probe import ProbeSettings

    recon_para = ReconParams(
        Data_Name="Move",
        Date="20260403",
        Probe="L15_80M_128",
        frame_num=200,
    )
    paths = PathConfig(data_root=r"E:\Study\PhD2\Data\Marsonics")
    para = ProbeSettings.from_json(paths.probe_settings_file(recon_para.Probe))
    run_pipeline(recon_para, para, paths)


if __name__ == "__main__":
    example_main()
