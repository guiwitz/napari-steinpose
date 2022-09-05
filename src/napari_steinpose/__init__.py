
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader_mcd
from .steinpose_widget import SteinposeWidget

__all__ = (
    "napari_get_reader_mcd",
    "SteinposeWidget",
)
