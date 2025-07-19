from gitingest import ingest

path_to_codebase = "./../src"
summary, tree, content = ingest(
    source=path_to_codebase,
    include_patterns={"*.py"},
    # exclude_patterns={"*.yaml", "*.txt", "__pycache__"},  # "*.sh"
)
with open("codebase.txt", "w", encoding="utf-8") as f:
    f.write(summary + "\n\n\n\n" + tree + "\n\n\n\n" + content)
