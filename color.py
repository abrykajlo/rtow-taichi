from vec3 import vec3

import taichi as ti

color = vec3

@ti.func
def write_color(image, width, height, i, j, pixel_color):
    image[i, height - 1 - j] = pixel_color