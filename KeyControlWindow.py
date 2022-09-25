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


def getDisplay(cap_src=None, surface_src=None, show_stat=False, stat=('___', '___', '___')):
    pygame.surfarray.blit_array(surface_src, setFrame(cap_src))
    if show_stat:
        battery, temp, flight_time = stat
        # text font
        font = pygame.font.SysFont("Adobe Hebrew", 20)

        b_title = font.render("Battery", True, "white")
        b_text = font.render(f'{battery}%', True, "white")

        t_title = font.render("Temp", True, "white")
        t_text = font.render(f'{temp}°C', True, "white")

        ft_title = font.render("Flight Time", True, "white")
        ft_text = font.render(f'{flight_time} sec', True, "white")

        # text center positions
        b_pos = b_title.get_rect(center=(30, 10))
        b_val_pos = b_text.get_rect(center=(33, 25))
        t_pos = b_title.get_rect(center=(30, 50))
        t_val_pos = b_text.get_rect(center=(33, 65))
        a_pos = b_title.get_rect(center=(30, 90))
        a_val_pos = b_text.get_rect(center=(33, 105))

        # display stat
        surface_src.blit(b_title, b_pos)
        surface_src.blit(b_text, b_val_pos)
        surface_src.blit(t_title, t_pos)
        surface_src.blit(t_text, t_val_pos)
        surface_src.blit(ft_title, a_pos)
        surface_src.blit(ft_text, a_val_pos)
    else:
        pass
    pygame.display.update()


def show_status(battery='___', temp='___', altitude='___'):
    return battery, temp, altitude


def main():
    keye = pygame.key.get_pressed().count(1)
    has = int(time.process_time()) % 2
    # print(int(time.process_time()) % 2)
    # print(has)
    if getKey("LEFT") and keye:
        if has == 0:
            print(has)
            print("Left key pressed {}".format(keye))
            time.sleep(0.03)
        else:
            pass

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
    # test code using pc webcam
    camera = cv.VideoCapture(0)
    rec = False
    camera.set(3, 640)
    camera.set(4, 480)
    init()
    screen = getFrame((640, 480), '  TELLO AI Live Stream')


    def record():
        # create a VideoWrite object, recording to ./video.avi

        height, width = 480, 640
        print(height)
        print(width)
        video = cv.VideoWriter(f'TELLO AI Videos/{time.strftime("VID%Y%m%d%I%M%S")}.avi',
                               cv.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        while rec:
            _, cap = camera.read()
            video.write(cap)
            time.sleep(1 / 30)
        video.release()


    # main()
    recorder = Thread(target=record)
    if not rec:
        pass
    else:
        recorder.start()
    try:
        while True:
            ret, img = camera.read()
            img = cv.resize(img, (640, 480))

            main()
            getDisplay(img, screen, True)
            cv.waitKey(1)
            if win_close():
                if not rec:
                    pass
                else:
                    recorder.join()
                sys.exit(0)
    except Exception as err:
        print(err)
        rec = False
        if not rec:
            pass
        else:
            recorder.join()
        pygame.quit()
        cv.destroyAllWindows()
        sys.exit(0)
