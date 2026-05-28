from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ProgressStep:
    key: str
    label: str
    weight: int


def build_progress_steps(recon_para) -> list[ProgressStep]:
    steps: list[ProgressStep] = []

    if recon_para.flag_Recon:
        steps.append(ProgressStep("recon_prepare", "Preparing reconstruction", 1))
        steps.append(ProgressStep("stream_pipeline", "Processing and reconstructing frames", recon_para.frame_num))

    if recon_para.flag_Video:
        steps.append(ProgressStep("video", "Generating video", recon_para.frame_num))

    if recon_para.flag_SVD:
        steps.append(ProgressStep("svd_load", "Loading SVD data", 1))
        steps.append(ProgressStep("svd_compute", "Computing SVD", 1))
        if recon_para.flag_SVD_Frames:
            steps.append(ProgressStep("svd_frames", "Saving SVD frames", recon_para.frame_num))
        if recon_para.flag_SVD_Video:
            steps.append(ProgressStep("svd_video", "Generating SVD video", recon_para.frame_num))
        steps.append(ProgressStep("svd_noise", "Estimating noise", 1))
        steps.append(ProgressStep("svd_figure", "Saving SVD figure", 1))

    steps.append(ProgressStep("completed", "Completed", 1))
    return steps


class PipelineProgressReporter:
    def __init__(self, recon_para, progress_callback=None, start_time=None):
        self.progress_callback = progress_callback
        self.start_time = time.perf_counter() if start_time is None else start_time
        self.steps = build_progress_steps(recon_para)
        self.stage_weights: dict[str, int] = {}
        self.stage_offsets: dict[str, int] = {}

        offset = 0
        for step in self.steps:
            weight = max(int(step.weight), 1)
            self.stage_offsets[step.key] = offset
            self.stage_weights[step.key] = weight
            offset += weight

        self.total_weight = max(offset, 1)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def emit(self, stage, current, total, message=None, indeterminate=False):
        if self.progress_callback is None:
            return

        stage = stage or "runtime"
        current_value = max(int(current or 0), 0)
        total_value = max(int(total or 0), 0)
        stage_weight = self.stage_weights.get(stage, max(total_value, 1) if total_value else 1)
        stage_offset = self.stage_offsets.get(stage, 0)

        fraction = 0.0
        if total_value > 0:
            fraction = min(max(current_value / total_value, 0.0), 1.0)
        elif current_value > 0 and not indeterminate:
            fraction = 1.0

        overall_current = min(stage_offset + fraction * stage_weight, self.total_weight)

        payload = {
            "stage": stage,
            "current": current_value,
            "total": total_value,
            "message": message or stage,
            "indeterminate": bool(indeterminate),
            "elapsed_seconds": self.elapsed_seconds(),
            "overall_current": overall_current,
            "overall_total": self.total_weight,
        }
        self.progress_callback(payload)

    def callback(self, default_stage=None, default_message=None):
        def _callback(payload):
            payload = payload or {}
            self.emit(
                payload.get("stage") or default_stage or "runtime",
                payload.get("current", 0),
                payload.get("total", 0),
                message=payload.get("message") or default_message or default_stage or "Running",
                indeterminate=payload.get("indeterminate", False),
            )

        return _callback
