from dataclasses import dataclass

@dataclass
class ArgParam:
    name: str
    val: (any, any)
