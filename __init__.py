"""
ComfyUI_Flux2ImageReference - Image Reference Node for Flux2-based Image Generation
"""

from .flux2_image_reference_node import NakuNode_Flux2

NODE_CLASS_MAPPINGS = {
    "NakuNode_Flux2": NakuNode_Flux2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NakuNode_Flux2": "NakuNode Flux2 Image Reference",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']