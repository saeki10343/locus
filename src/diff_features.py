# src/diff_features.py
import re

FUNCTION_REGEX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
SYNTAX_KEYWORDS = ["if", "for", "while", "switch", "case", "try", "catch"]


def extract_features_from_patch(patch: str, weight: int = 1) -> str:
    """Extract simple features from a diff patch.

    Parameters
    ----------
    patch : str
        Unified diff patch text.

    Returns
    -------
    str
        Space separated tokens representing function names and syntax keywords
        found in the patch.
    """
    tokens = []
    for line in patch.splitlines():
        if line.startswith("+") or line.startswith("-"):
            # ignore diff metadata like +++/---
            if line.startswith("+++") or line.startswith("---"):
                continue
            func_names = FUNCTION_REGEX.findall(line)
            tokens.extend(func_names)
            for kw in SYNTAX_KEYWORDS:
                if re.search(rf"\b{kw}\b", line):
                    tokens.append(f"kw_{kw}")
        elif line.startswith("@@"):
            # hunk header may include function signature context
            context = line.strip("@ ")
            func_names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", context)
            tokens.extend(func_names)
    weighted = []
    for tok in tokens:
        weighted.extend([tok] * weight)
    return " ".join(weighted)
