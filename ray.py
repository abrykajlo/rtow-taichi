from vec3 import vec3, point3

import taichi as ti

@ti.dataclass
class ray:
    origin: point3
    direction: vec3

    def __init__(self, origin, direction) -> None:
        self.origin = origin
        self.direction = direction
    
    @ti.func
    def at(self, t):
        return self.origin + t * self.direction