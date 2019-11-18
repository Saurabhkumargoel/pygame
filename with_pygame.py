import pygame
import pygame.camera
# from pygame.locals import *

pygame.init()
pygame.camera.init()


# cam = pygame.camera.Camera("/dev/video0",(640,480))
# cam.start()
# image = cam.get_image()
class Capture(object):
    def __init__(self):
        self.size = (640,480)
        # create a display surface. standard pygame stuff
        self.display = pygame.display.set_mode(self.size, 0)

        # this is the same as what we saw before
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")
        self.cam = pygame.camera.Camera(self.clist[0], self.size)
        self.cam.start()

        # create a surface to capture to.  for performance purposes
        # bit depth is the same as that of the display surface.
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_and_flip(self):
        # if you don't want to tie the framerate to the camera, you can check
        # if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        if self.cam.query_image():
            self.snapshot = self.cam.get_image(self.snapshot)

        # blit it to the display surface.  simple!
        self.display.blit(self.snapshot, (0,0))
        pygame.display.flip()

    def main(self):
        going = True
        while going:
            events = pygame.event.get()
            for e in events:
                # if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                if e.type == 0xFF:
                    # close the camera safely
                    self.cam.stop()
                    going = False

            self.get_and_flip()


c = Capture()
c.main()

# import cv2
# input_stream = cv2.VideoCapture('/dev/video0')


# while True:
# 	i,frame = input_stream.read()
# 	cv2.imshow('test',frame)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break


# input_stream.release()
# cv2.destroyAllWindows()