"""ComfyUI-P-H3ImageLookSheets — look/outfit reference shots from P-H3-Image.

Self-contained: no imports outside the standard library and what ComfyUI
already provides.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
