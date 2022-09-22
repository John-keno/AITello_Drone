import pygame
import cv2 as cv
import sys
import time
from threading import Thread


def init():
    pygame.init()


def getFrame(window_size=(None, None), window_name='display window'):
    icon = pygame.image.load('UI/drone.png')
    pygame.display.set_icon(icon)
    win = pygame.display.set_mode(window_size)
    pygame.display.set_caption(window_name)
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
    pygame.draw.rect(surface_src, (255, 0, 0), (20, 25, 20, 5))
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
    # test code
    camera = cv.VideoCapture(0)
    rec = True
    camera.set(3, 640)
    camera.set(4, 480)
    init()
    screen = getFrame((640, 480), '  TELLO AI Live Stream')


    # # main()

    def record():
        # create a VideoWrite object, recording to ./video.avi

        height, width = 480, 640
        print(height)
        print(width)
        video = cv.VideoWriter(f'TELLO AI Videos/{time.strftime("VID%Y%m%d%I%M%S")}.avi',
                               cv.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        while rec:
            camera.read()
            video.write(frame)
            time.sleep(1 / 30)

        video.release()


    # main()
    recorder = Thread(target=record)
    recorder.start()
    try:
        while True:
            ret, frame = camera.read()
            frame = cv.resize(frame, (640, 480))
            main()
            pygame.draw.rect(screen, (255, 0, 0), (200, 250, 200, 5))
            getDisplay(frame, screen)
            cv.waitKey(1)
            if win_close():
                rec = False
                recorder.join()
                sys.exit(0)
    except Exception as err:
        print(err)
        rec = False
        recorder.join()
        pygame.quit()
        cv.destroyAllWindows()
        sys.exit(0)
