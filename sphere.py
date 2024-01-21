from hittable import MISSED_RECORD, hittable_type
from vec3 import point3

import taichi as ti
import taichi.math as tm

sphere_type = hittable_type()

@ti.dataclass
class Sphere:
    center: point3
    radius: ti.f64
    
    @ti.func
    def hit(self, r, ray_tmin, ray_tmax):
        rec = MISSED_RECORD
        oc = r.origin - self.center
        a = tm.dot(r.direction, r.direction)
        half_b = tm.dot(oc, r.direction)
        c = tm.dot(oc, oc) - self.radius * self.radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            rec.t = -1
        else:
            sqrtd = tm.sqrt(discriminant)

            # Find the nearest root that lies in the acceptable range
            rec.t = root = (-half_b - sqrtd) / a
            if root <= ray_tmin or ray_tmax <= root:
                rec.t = root = (-half_b + sqrtd) / a
                if root <= ray_tmin or ray_tmax <= root:
                    rec.t = -1
        
        if rec.hit():
            rec.p = r.at(rec.t)
            rec.normal = (rec.p - self.center) / self.radius
        return rec