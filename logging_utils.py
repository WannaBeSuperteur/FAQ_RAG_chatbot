

import json
import datetime


LOGGING_FILE_PATH = 'logging.txt'


def add_log(tag:str, case_id:int, content:str, logging_file_path:str=LOGGING_FILE_PATH) -> None:
    item = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "log_case_id": case_id,
        "tag": tag,
        "content": content,
    }
    with open(logging_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
