
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader_mcd
from .mcd_widget import MCDWidget

__all__ = (
    "napari_get_reader_mcd",
    "MCDWidget",
)
