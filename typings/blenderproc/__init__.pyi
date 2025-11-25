# Minimal stub so Pyright doesn't descend into the real BlenderProc package.
# Runtime uses the actual package; type checking only needs the symbol to exist.
from typing import Any

def init(*args: Any, **kwargs: Any) -> Any: ...
