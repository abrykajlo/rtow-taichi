from vec3 import vec3, point3

import taichi as ti

next_type = 0
def hittable_type():
    global next_type
    type = next_type
    next_type += 1
    return type

@ti.dataclass
class Hittable:
    type: int
    index: int

@ti.dataclass
class HitRecord:
    p: point3
    normal: vec3
    t: ti.f64

    @ti.func
    def hit(self):
        return self.t > 0

MISSED_RECORD = HitRecord(t=-1)