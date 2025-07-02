from dataclasses import dataclass

@dataclass
class IMetadata:
    source: str
    title: str
    docid: str