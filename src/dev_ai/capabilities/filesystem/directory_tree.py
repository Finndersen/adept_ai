import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

logger = logging.getLogger(__name__)


@dataclass
class FileSystemItem:
    path: Path
    is_directory: bool
    # None if it is a file or unexpanded directory
    children: list["FileSystemItem"] | None
    # Whether the children of the directory are displayed in directory tree
    expanded: bool = False

    @property
    def name(self) -> str:
        """Get the name of the file or directory."""
        return self.path.name


class DirectoryTree:
    """
    Class that holds a directory tree and provides methods for formatting it as a tree or list of paths.
    """

    root_directory: Path
    initial_directory_depth: int
    root_item: FileSystemItem
    git_repo: Repo | None
    _tree_cache: str | None
    _paths_cache: str | None

    def __init__(
        self, root_directory: str | Path | None = None, max_directory_depth: int = 3, respect_gitignore: bool = True
    ):
        self.root_directory = (Path(root_directory) if root_directory else Path.cwd()).resolve()
        # Set up git repo and gitignore handling if needed
        self.git_repo = None
        if respect_gitignore:
            try:
                # Try to find the git repository containing this path
                self.git_repo = Repo(self.root_directory, search_parent_directories=True)
                logger.debug(f"Respecting .gitignore rules from repository at {self.git_repo.working_dir}")
            except (InvalidGitRepositoryError, NoSuchPathError):
                # Not in a git repository or path doesn't exist
                pass
        self.initial_directory_depth = max_directory_depth
        self.root_item = self._build_directory_tree(self.root_directory)
        self._tree_cache = None
        self._paths_cache = None

    def expand_directory(self, target_path: Path) -> bool:
        """
        Expand a directory in the tree by its path.
        Returns True if the directory was found and expanded, False otherwise.
        """

        def _expand_directory(item: FileSystemItem, target: Path, current_depth: int = 0) -> bool:
            # Quick check: if this item's path is not a parent of the target path,
            # we can skip this entire branch
            try:
                target.relative_to(item.path)
            except ValueError:
                return False

            # If we've reached the target path, expand this directory
            if item.path == target:
                # Rebuild the directory tree from this point
                item.children = self._build_directory_tree(item.path, current_depth).children
                item.expanded = True
                return True

            # Recursively search children if they exist
            if item.children:
                for child in item.children:
                    if _expand_directory(child, target, current_depth + 1):
                        return True
            return False

        result = _expand_directory(self.root_item, target_path)
        self._clear_display_caches()
        return result

    def _build_directory_tree(self, dir_path: Path, depth: int = 0) -> FileSystemItem:
        """Build a directory tree from the given path."""
        dir_entires = self._get_filtered_dir_entries(dir_path)

        # Sort items by directories first, then files, then alphabetically
        children = []
        for entry_path in sorted(dir_entires, key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip .gitignored or other other files
            if entry_path.name == ".git":
                continue

            if entry_path.is_dir():
                if depth < self.initial_directory_depth:
                    children.append(self._build_directory_tree(entry_path, depth + 1))
                else:
                    children.append(FileSystemItem(path=entry_path, children=None, expanded=False, is_directory=True))
            else:
                children.append(FileSystemItem(path=entry_path, children=None, is_directory=False))

        return FileSystemItem(path=dir_path, children=children, expanded=True, is_directory=True)

    def _get_filtered_dir_entries(self, dir_path: Path) -> Iterable[Path]:
        """Get the entries of a directory, filtered by gitignore rules."""
        dir_entires = dir_path.iterdir()

        if not self.git_repo:
            return dir_entires

        rel_entries = [(entry, str(entry.relative_to(self.git_repo.working_dir))) for entry in dir_entires]
        ignored_entries = set(self.git_repo.ignored(*[entry[1] for entry in rel_entries]))
        return [entry[0] for entry in rel_entries if entry[1] not in ignored_entries]

    def format_as_tree(self) -> str:
        if not self._tree_cache:
            self._tree_cache = "\n".join(self._format_tree(self.root_item))

        return self._tree_cache

    def _format_tree(self, item: FileSystemItem, prefix: str = "", is_root: bool = True) -> List[str]:
        lines = []

        # Skip adding the root item name at the start if it's not the root
        if is_root:
            name = item.name + "/" if item.is_directory else item.name
            lines.append(f"{prefix}{name}")
            prefix = ""  # Reset prefix for children of root

        if item.children:
            items = item.children
            for i, child in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = prefix + ("└─ " if is_last else "├─ ")

                # Format the child name
                if child.is_directory:
                    name = child.name + "/"
                    if not child.expanded:
                        name += " [not expanded]"
                else:
                    name = child.name

                lines.append(f"{current_prefix}{name}")

                # Recursively format children if directory is expanded
                if child.is_directory and child.expanded and child.children:
                    next_prefix = prefix + ("│  " if not is_last else "   ")
                    lines.extend(self._format_tree(child, next_prefix, is_root=False))

        return lines

    def _clear_display_caches(self):
        self._tree_cache = None
        self._paths_cache = None

    def format_as_paths(self) -> str:
        """
        Format the directory tree as a list of relative paths.
        Each path is relative to the root directory.
        """
        if not self._paths_cache:
            self._paths_cache = "\n".join(self._collect_paths(self.root_item))
        return self._paths_cache

    def _collect_paths(self, item: FileSystemItem) -> List[str]:
        paths = []

        if item.path != self.root_directory:
            # Add the current item's path
            rel_path = str(item.path.relative_to(self.root_directory))
            if item.is_directory:
                rel_path += "/"
                if not item.expanded:
                    rel_path += " [not expanded]"

            paths.append(rel_path)

        # Add children's paths
        if item.children:
            for child in item.children:
                paths.extend(self._collect_paths(child))

        return paths
