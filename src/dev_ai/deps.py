from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentDeps:
    current_working_directory: Path
