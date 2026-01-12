from enum import Enum
from typing import Tuple, Union

class MarkerType(Enum):
    NONE = "none"
    DOT = "dot"
    CIRCLE = "circle" 
    SQUARE = "square"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    CROSS = "cross"

class MarkerStyle:

    def __init__(
        self,
        marker_type: Union[MarkerType, str] = MarkerType.CIRCLE,
        color: Tuple[int, int, int] = (255, 0, 0),
        radius: int = 5,
        thickness: int = 2,
        show_caption: bool = True,
        font_scale: float = 0.5,
        text_offset: Tuple[int, int] = (0, -25),
        text_bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_bold: bool = False,
        font_italic: bool = False,
        text_label_override: str = None,
    ):
        self.marker_type = marker_type if isinstance(marker_type, MarkerType) else MarkerType(marker_type)
        self.color = color
        self.radius = radius
        self.thickness = thickness
        self.show_caption = show_caption
        self.font_scale = font_scale
        self.text_offset = text_offset
        self.text_bg_color = text_bg_color
        self.text_color = text_color
        self.font_bold = font_bold
        self.font_italic = font_italic
        self.text_label_override = text_label_override

    def __repr__(self):
        return f"DepthStyle(marker_type={self.marker_type}, color={self.color}, radius={self.radius}, thickness={self.thickness}, show_caption={self.show_caption}, font_scale={self.font_scale}, text_offset={self.text_offset}, text_bg_color={self.text_bg_color}, text_color={self.text_color}, font_bold={self.font_bold}, font_italic={self.font_italic}, text_label_override={self.text_label_override})"

class DepthStyle(MarkerStyle):
    pass

class SemanticCorrespondenceStyle(MarkerStyle):
    pass
