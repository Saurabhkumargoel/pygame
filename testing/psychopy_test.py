import ratcave as rc
from psychopy import visual, event

# Create Window
window = visual.Window()

# Insert filename into WavefrontReader.
obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)

# Create Mesh
monkey = obj_reader.get_mesh("Monkey")
monkey.position.xyz = 0, 0, -2

# Create Scene
scene = rc.Scene(meshes=[monkey])

while 'escape' not in event.getKeys():
    with rc.default_shader:
        scene.draw()
    window.flip()

window.close()