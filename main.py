import taichi as ti

ti.init(arch=ti.gpu)

image_width = 256
image_height = 256

image = ti.Vector.field(3, ti.f32, shape=(image_width, image_height))

@ti.func
def paint(i, j, r, g, b):
    image[i, image_height - 1 - j] = ti.Vector([r, g, b])

@ti.kernel
def render():
    for i, j in image:
        r = i / (image_width - 1)
        g = j / (image_height - 1)
        b = 0

        paint(i, j, r, g, b)

def main():
    gui = ti.GUI('Ray Tracing in One Weekend', res=(image_width, image_height))
    render()
    while gui.running:
        gui.set_image(image)
        gui.show()


if __name__ == '__main__':
    main()