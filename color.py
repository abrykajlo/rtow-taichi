from vec3 import vec3

import taichi as ti

color = vec3

@ti.func
def write_color(image, i, j, pixel_color):
    image[i, j] = pixel_color