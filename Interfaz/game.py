import pygame
import sys

# Initialize Pygame
pygame.init()
pygame.font.init()

# Configurable Variables
WINDOW_WIDTH, WINDOW_HEIGHT = 1900, 1000
scroll_list_area_size = (1900, 350)
current_text_area_size = (1900, 120)
keyboard_layout_area_size = (1900, 550)
button_size = (180, 115)
keyboard_row_offsets = [-10, 35, 75, 170]
spacebar_width_multiplier = 6
item_height = 80
list_font_size = 48
font = pygame.font.SysFont('Stark', list_font_size)

# Define colors
TRANSLUCENT_WHITE = (255, 255, 255, 128)  # The last number is the alpha value
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (73, 73, 73)
PURPLE = (206, 130, 242)
CYAN = (163, 237, 216)
GREY_BACKGROUND = (31, 31, 31)
BLUE_TRON = (121, 245, 246)
BLUE = (19, 32, 199)
LIGHT_BLUE = (5, 187, 247)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Customizable Interface")

background_image = pygame.image.load('Fondo.png').convert()
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))

overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
overlay.fill((50, 50, 50, 240))

# Define areas
scroll_list_area = pygame.Rect(0, 0, *scroll_list_area_size)
current_text_area = pygame.Rect(0, scroll_list_area.bottom, *current_text_area_size)
keyboard_layout_area = pygame.Rect(0, current_text_area.bottom + 50, *keyboard_layout_area_size)

# Keyboard layout setup
keyboard_rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM9", " "]
first_key_x = keyboard_layout_area.x + 50
first_key_y = keyboard_layout_area.y + 50

# Scrollable list setup
list_items = ["Variable1", "Variable2", "Variable3", "Variable4"]
scroll_y = 0
list_font = pygame.font.SysFont(None, list_font_size)

# Initialize variables for text input and cursor
input_text = ''
cursor = '|'
cursor_visible = True
cursor_switch_ms = 500
last_switch = pygame.time.get_ticks()

def draw_keyboard():
    for i, row in enumerate(keyboard_rows):
        # Start position for each row (adjusted by the row's offset)
        x = first_key_x + keyboard_row_offsets[i]

        for j, key in enumerate(row):
            # Determine the width of the key
            key_width = button_size[0]
            if key == " ":
                key_width = button_size[0] * spacebar_width_multiplier

            # Create the rectangle for the key
            key_rect = pygame.Rect(x, first_key_y + i * (button_size[1] + 5), key_width, button_size[1])

            # Draw the key rectangle
            pygame.draw.rect(screen, GREY_BACKGROUND, key_rect)
            pygame.draw.rect(screen, LIGHT_BLUE, key_rect, 2)  # The border

            # Set the text for the key (change '9' to 'Enter')
            display_text = 'Enter' if key == '9' else key

            # Render and center
             # Render and center the text within the key rectangle
            text_surface = font.render(display_text, True, BLUE_TRON)
            text_rect = text_surface.get_rect(center=key_rect.center)
            screen.blit(text_surface, text_rect)

            # Move x position to the next key's position
            x += key_rect.width + 5  # Adding 5 pixels for spacing between keys


def draw_scroll_list():
    pygame.draw.rect(screen, CYAN, scroll_list_area, 2)
    for index, item_text in enumerate(list_items):
        item_y = index * item_height + scroll_y
        item_rect = pygame.Rect(scroll_list_area.left, item_y, scroll_list_area.width, item_height)
        if item_rect.colliderect(scroll_list_area):
            text_surface = font.render(item_text, True, BLUE_TRON)
            text_rect = text_surface.get_rect(center=(scroll_list_area.centerx, item_rect.centery))
            if scroll_list_area.top <= text_rect.top and scroll_list_area.bottom >= text_rect.bottom:
                pygame.draw.rect(screen, GRAY, item_rect)
                screen.blit(text_surface, text_rect)

def draw_interface():
    screen.fill(BLACK)
    draw_keyboard()
    draw_scroll_list()
    pygame.draw.rect(screen, GREY_BACKGROUND, current_text_area)
    text_surface = font.render(input_text + (cursor if cursor_visible else ''), True, BLUE_TRON)
    screen.blit(text_surface, (current_text_area.x+10, current_text_area.y+10))

def process_computer_vision_signals():
    # [Placeholder for computer vision signal processing]
    pass

# Game loop
running = True
while running:
    current_time = pygame.time.get_ticks()

    if current_time - last_switch > cursor_switch_ms:
        cursor_visible = not cursor_visible
        last_switch = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    process_computer_vision_signals()
    draw_interface()
    pygame.display.flip()

pygame.quit()
sys.exit()