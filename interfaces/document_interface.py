from dataclasses import dataclass
from interfaces.metadata_interface import IMetadata

@dataclass
class IDocument:
    text: str
    distance: float
    metadata: IMetadata