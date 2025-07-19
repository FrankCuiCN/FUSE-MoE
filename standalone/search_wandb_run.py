import os, sys

root = "projects/moe/results"                  # change if your root differs
target = input("wandb job id? ").strip()

stack = [root]                                 # simple DFS with a stack
while stack:
    current = stack.pop()
    try:
        entries = [os.path.join(current, e) for e in os.listdir(current)]
    except (FileNotFoundError, PermissionError):
        continue                               # skip unreadable dirs

    for e in entries:
        if os.path.isdir(e):
            stack.append(e)                    # push sub-directories
        elif os.path.basename(e) == "wand_job_id.txt":
            try:
                if target in open(e).read():
                    print("Found in:", os.path.abspath(e))
                    sys.exit(0)
            except Exception:
                pass                           # ignore unreadable files
print("Not found.")
