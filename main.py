import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, default_fp=ti.f64)

from color import color, write_color
from hittable import HitRecord
from world import World
from ray import ray
from sphere import Sphere
from vec3 import point3, vec3


# Image

aspect_ratio = 16 / 9
image_width = 400

# Calculate the image height, and ensure that it's at least 1.
image_height = image_width / aspect_ratio
image_height = 1 if image_height < 1 else image_height
image_height = int(image_height)

# World

world = World()

world.add(Sphere(point3([0, 0, -1]), 0.5))
world.add(Sphere(point3([0, -100.5, -1]), 100))
world.lock()

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

WHITE = color([1, 1, 1])
BLUE = color([0.5, 0.7, 1.0])

@ti.func
def ray_color(r: ray, world: ti.template()):
    result = color([0, 0, 0])

    rec = world.hit(r, 0, tm.inf)
    if rec.hit():
        result = 0.5 * (rec.normal + color([1, 1, 1]))
    else:
        unit_direction = tm.normalize(r.direction)
        a = 0.5 * (unit_direction.y + 1)
        result = (1 - a) * WHITE + a * BLUE
    return result

@ti.kernel
def render():
    for i, j in image:
        pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
        ray_direction = pixel_center - camera_center
        r = ray(camera_center, ray_direction)
        
        pixel_color = ray_color(r, world)
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