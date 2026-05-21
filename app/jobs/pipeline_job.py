import sys
import traceback

from app.progress_protocol import encode_progress
from pipeline.runner import run_pipeline
from utils.runtime_io import load_runtime_bundle


def run_from_bundle(config_path: str):
    paths, recon_para, probe = load_runtime_bundle(config_path)
    run_pipeline(
        recon_para,
        probe,
        paths,
        use_tqdm=False,
        logger=print,
        progress_callback=lambda payload: print(encode_progress(payload)),
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline_job.py <config.json>")
        sys.exit(2)

    try:
        run_from_bundle(sys.argv[1])
    except Exception as exc:
        print(f"Run failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
