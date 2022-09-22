import pygame
import cv2 as cv
import sys


def init():
    pygame.init()


def getFrame(frame_size=(None, None)):
    win = pygame.display.set_mode(frame_size)
    return win


def setFrame(frame_src=None):
    frame_src = cv.cvtColor(frame_src, cv.COLOR_BGR2RGB)
    frame_src = frame_src.swapaxes(0, 1)
    return frame_src


def getKey(keyName):
    ans = False
    for eve in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    # print('K_{}'.format(keyName))

    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


def getDisplay(cap_src=None, surface_src=None):
    pygame.surfarray.blit_array(surface_src, setFrame(cap_src))
    pygame.display.update()


# def quit_display
def main():
    if getKey("LEFT"):
        print("Left key pressed")

    if getKey("RIGHT"):
        print("Right key Pressed")


def win_close():
    close = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            close = True
    return close


if __name__ == "__main__":
    camera = cv.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    init()
    pygame.display.set_caption("OpenCV camera stream on Pygame")
    screen = getFrame((1280, 720))

    # # main()
    try:
        while True:
            ret, frame = camera.read()
            frame = cv.resize(frame, (1280, 720))
            main()
            getDisplay(frame, screen)
            cv.waitKey(1)
            if win_close():
                sys.exit(0)
    except:
        pygame.quit()
        cv.destroyAllWindows()
