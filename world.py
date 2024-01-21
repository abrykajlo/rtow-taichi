from hittable import Hittable, HitRecord
from sphere import Sphere, sphere_type

import taichi as ti

class World:
    locked: bool

    def __init__(self) -> None:
        self.locked = False
        self.hittable_list = []
        self.spheres = []
    
    def add(self, hittable):
        if self.locked:
            raise Exception('Can\'t add to locked world')
        if isinstance(hittable, Sphere):
            self.hittable_list.append(Hittable(sphere_type, len(self.spheres)))
            self.spheres.append(hittable)
        else:
            raise Exception('Hittable does not exist')

    def lock(self):
        # convert hittable list to ndarray
        self.hittable_count = len(self.hittable_list)
        hittable_list = Hittable.field(shape=len(self.hittable_list))
        for i, item in enumerate(self.hittable_list):
            hittable_list[i] = item
        self.hittable_list = hittable_list

        # convert spheres to ndarray
        spheres = Sphere.field(shape=len(self.spheres))
        for i, item in enumerate(self.spheres):
            spheres[i] = item
        self.spheres = spheres

        self.locked = True

    @ti.func
    def hit(self, r, ray_tmin, ray_tmax):
        rec = HitRecord()
        closest_so_far = ray_tmax

        for i in range(self.hittable_count):
            temp_rec = HitRecord()
            hittable = self.hittable_list[i]
            if hittable.type == sphere_type:
                temp_rec = self.spheres[hittable.index].hit(r, ray_tmin, closest_so_far)
            if temp_rec.hit():
                rec = temp_rec
                closest_so_far = rec.t

        return rec