import json


PROGRESS_PREFIX = "__PROGRESS__ "


def encode_progress(payload: dict) -> str:
    return f"{PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=False)}"


def parse_progress_line(line: str):
    if not line.startswith(PROGRESS_PREFIX):
        return None
    payload = line[len(PROGRESS_PREFIX):].strip()
    if not payload:
        return None
    return json.loads(payload)
