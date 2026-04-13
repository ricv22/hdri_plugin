"""
ComfyUI custom nodes: GMNet inverse tone mapping (gain-map HDR reconstruction).

Requires a local clone of https://github.com/qtlark/GMNet and pretrained weights.
Set GMNET_CODES_ROOT to GMNet's ``codes`` directory (the folder that contains ``models/``).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
