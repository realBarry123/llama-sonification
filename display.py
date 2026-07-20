import pygame
import sounddevice as sd
import threading
import queue
import config
from sonifier import Sonifier
from model_wrapper import ModelWrapper

FULL_SCREEN = config.FULL_SCREEN

# MODEL SETUP

sonifier = Sonifier((1, 17, 2048), fs=config.FS)
model = ModelWrapper()
model.seed(config.SEED)


# PRODUCER
# producer-consumer pattern (very nice)

audio_queue = queue.Queue(maxsize=2)

def producer():
    while True:
        token, states, context = model.next()
        audio = sonifier(states)
        audio_queue.put((token, audio, context))  # blocks if queue is full

producer_thread = threading.Thread(target=producer, daemon=True)
producer_thread.start()


# GRAPHICS SETUP

pygame.init()
VW = config.VW
VH = config.VH
INIT_TEXT_COLOR = 255
INIT_D_TEXT_COLOR = -5
DD_TEXT_COLOR = 0.05

screen = pygame.display.set_mode((VW, VH))
clock = pygame.time.Clock()
if FULL_SCREEN: 
    pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
running = True
font = pygame.font.SysFont('Arial', 128)
context_font = pygame.font.SysFont('Arial', 20)
current_token = ""
context = ""
text_color = INIT_TEXT_COLOR
d_text_color = INIT_D_TEXT_COLOR

while running:

    # EVENTS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            if config.WRITE_HISTORY: 
                model.write_history("context.txt")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LSHIFT] and keys[pygame.K_9] and keys[pygame.K_SEMICOLON]:
        running = False
        if config.WRITE_HISTORY: 
            model.write_history("context.txt")
    elif keys[pygame.K_ESCAPE]:
        if config.WRITE_HISTORY: 
            model.write_history("context.txt")
        model.__init__()
        model.seed("")


    # CONSUMER (very nice)

    is_playing = False  # placeholder

    try:
        is_playing = sd.get_stream().active

    except RuntimeError:
        is_playing = False

    if not is_playing and not audio_queue.empty():
        current_token, audio, context = audio_queue.get()
        sd.play(audio, samplerate=config.FS)
        text_color = INIT_TEXT_COLOR
        d_text_color = INIT_D_TEXT_COLOR

    screen.fill("black")

    cur_surface = font.render(current_token, True, (text_color, text_color, text_color))
    text_rect = cur_surface.get_rect()
    text_rect.center = (VW // 2, VH // 2)
    screen.blit(cur_surface, text_rect)

    context_surface = context_font.render(context, True, (200, 200, 255))
    context_rect = context_surface.get_rect()
    context_rect.center = (VW // 2, 50)
    screen.blit(context_surface, context_rect)

    pygame.display.flip()

    d_text_color = min(d_text_color + DD_TEXT_COLOR, 0)
    text_color = max(text_color + d_text_color, 0)

    clock.tick(60)  # limits FPS to 60

pygame.quit()