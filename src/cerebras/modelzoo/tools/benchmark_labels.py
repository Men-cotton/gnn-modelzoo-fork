"""Stable label values accepted by the Cerebras SDK."""

import hashlib
from pathlib import Path
import re


def label_value(value: object, *, max_length: int = 63) -> str:
    """Bound an SDK label, reserving a hash only for unsafe or oversized values."""
    if not 12 <= max_length <= 63:
        raise ValueError("Label length must be between 12 and 63")
    original = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", original).strip("-_.") or "value"
    if cleaned != original or len(cleaned) > max_length:
        suffix = hashlib.sha256(original.encode()).hexdigest()[:10]
        cleaned = f"{cleaned[: max_length - 11].rstrip('-_.')}-{suffix}"
    return cleaned


def study_label(study_dir: Path, *, compact: bool = False) -> str:
    """Distinguish identically named studies in different directories."""
    path = Path(study_dir).resolve()
    suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
    if compact:
        return suffix
    return label_value(f"{path.name or 'study'}-{suffix}")
