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
viewport_v = vec3([0, -viewport_height, 0])

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height

# Calculate the location of the upper left pixel.
viewport_upper_left = camera_center - vec3([0, 0, focal_length]) - viewport_u / 2 - viewport_v / 2
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

image_shape = (image_width, image_height)
image = ti.Vector.field(3, ti.f32, shape=image_shape)

@ti.func
def ray_color(r: ray):
    unit_direction = tm.normalize(r.direction)
    a = 0.5 * (unit_direction.y + 1)
    return (1 - a) * color([1, 1, 1]) + a * color([0.5, 0.7, 1.0])

@ti.kernel
def render():
    for i, j in image:
        pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
        ray_direction = pixel_center - camera_center
        r = ray(camera_center, ray_direction)

        pixel_color = ray_color(r)
        write_color(image, image_width, image_height, i, j, pixel_color)

def main():
    gui = ti.GUI('Ray Tracing in One Weekend', res=image_shape)
    render()
    while gui.running:
        gui.set_image(image)
        gui.show()


if __name__ == '__main__':
    main()