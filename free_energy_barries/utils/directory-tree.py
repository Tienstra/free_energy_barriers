from pathlib import Path
import os


def generate_tree(startpath, exclude_patterns=None):
    """
    Generate a directory tree structure in ASCII.

    Args:
        startpath (str): Root directory to start from
        exclude_patterns (list): List of patterns to exclude (e.g., ['__pycache__', '*.pyc'])
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".git",
            ".pytest_cache",
        ]

    def should_exclude(path):
        path_str = str(path)
        return any(pattern in path_str for pattern in exclude_patterns)

    output = []
    root_dir = Path(startpath)

    def add_dir(dir_path, prefix=""):
        contents = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        pointers = [
            "├── " if i < len(contents) - 1 else "└── " for i in range(len(contents))
        ]

        for pointer, path in zip(pointers, contents):
            if should_exclude(path):
                continue

            output.append(f"{prefix}{pointer}{path.name}")

            if path.is_dir():
                extension = "│   " if pointer == "├── " else "    "
                add_dir(path, prefix=prefix + extension)

    output.append(root_dir.name)
    add_dir(root_dir)

    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    tree = generate_tree(".")
    print(tree)
