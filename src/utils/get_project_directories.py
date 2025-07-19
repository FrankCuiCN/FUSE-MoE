import os
import re


def get_project_directories(
    project_type, config, is_main, subdirectories
) -> tuple[str, list]:
    project_dir = os.path.join(
        config.misc_project_directory, config.misc_run_name
    )
    os.makedirs(project_dir, exist_ok=True)
    pattern = re.compile(rf"{project_type} \((\d+)\)")
    dirs = [d for d in os.listdir(project_dir) if pattern.match(d)]
    if is_main:
        # Main process creates a new run directory.
        run_number = (
            1
            if not dirs
            else max(int(pattern.match(d).group(1)) for d in dirs) + 1
        )
    else:
        # Other processes use the most recently created directory.
        run_number = (
            1
            if not dirs
            else max(int(pattern.match(d).group(1)) for d in dirs)
        )
    RUN_NAME = f"{project_type} ({run_number})"
    RESULT_PATH = os.path.join(project_dir, RUN_NAME)
    subdirectory_paths = []
    for i in subdirectories:
        item_path = os.path.join(RESULT_PATH, i)
        if is_main:
            os.makedirs(item_path, exist_ok=True)
        subdirectory_paths.append(item_path)
    # Q: Why not use "if is_main:" here?
    # A: throughput and vram script will fail if we check is_main here
    #   Essentially, we are handling is_main outside of get_project_directories()
    os.makedirs(RESULT_PATH, exist_ok=True)
    return RESULT_PATH, subdirectory_paths
