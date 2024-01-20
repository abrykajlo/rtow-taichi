from color import color, write_color
from ray import ray
from vec3 import point3, vec3

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

# Image

aspect_ratio = 16 / 9
image_width = 400

# Calculate the image height, and ensure that it's at least 1.
image_height = image_width / aspect_ratio
image_height = 1 if image_height < 1 else image_height
image_height = int(image_height)

# Camera

focal_length = 1
viewport_height = 2
viewport_width = viewport_height * image_width / image_height
camera_center = point3([0, 0, 0])

# Calculate the vectors across the horizontal and down the vertical viewport edges.
viewport_u = vec3([viewport_width, 0, 0])
viewport_v = vec3([0, viewport_height, 0])

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height

# Calculate the location of the upper left pixel.
viewport_bottom_left = camera_center - vec3([0, 0, focal_length]) - viewport_u / 2 - viewport_v / 2
pixel00_loc = viewport_bottom_left + 0.5 * (pixel_delta_u + pixel_delta_v)

image_shape = (image_width, image_height)
image = ti.Vector.field(3, ti.f32, shape=image_shape)

@ti.func
def hit_sphere(center, radius, r: ray):
    oc = r.origin - center
    a = tm.dot(r.direction, r.direction)
    b = 2 * tm.dot(oc, r.direction)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    return discriminant >= 0

WHITE = color([1, 1, 1])
BLUE = color([0.5, 0.7, 1.0])

@ti.func
def ray_color(r: ray):
    pixel_color = color([0, 0, 0])
    if hit_sphere(point3([0, 0, -1]), 0.5, r):
        pixel_color = color([1, 0, 0])
    else:
        unit_direction = tm.normalize(r.direction)
        a = 0.5 * (unit_direction.y + 1)
        pixel_color = (1 - a) * WHITE + a * BLUE
    return pixel_color

@ti.kernel
def render():
    for i, j in image:
        pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
        ray_direction = pixel_center - camera_center
        r = ray(camera_center, ray_direction)
        
        
        pixel_color = ray_color(r)
        write_color(image, i, j, pixel_color)

def main():
    gui = ti.GUI('Ray Tracing in One Weekend', res=image_shape)
    render()
    while gui.running:
        gui.set_image(image)
        gui.show()
    gui.set_image(image)
    gui.show('out.png')


if __name__ == '__main__':
    main()