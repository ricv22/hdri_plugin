"""
ComfyUI custom nodes: GMNet inverse tone mapping (gain-map HDR reconstruction).

Requires a local clone of https://github.com/qtlark/GMNet and pretrained weights.
Optional: set ``GMNET_CODES_ROOT`` / ``GMNET_CHECKPOINT`` (or ``GMNET_REPO_ROOT``).
If omitted, ``custom_nodes/GMNet/codes`` next to this package is used when present (e.g. RunComfy).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
