import pygame
import sys
import math

# Initialize Pygame
pygame.init()
pygame.font.init()

# Configurable Variables
WINDOW_WIDTH, WINDOW_HEIGHT = 1900, 1000
scroll_list_area_size = (1900, 350)
current_text_area_size = (1900, 120)
keyboard_layout_area_size = (1900, 550)
square_size = 400
button_size = (180, 115)
keyboard_row_offsets = [-10, 35, 75, 170]
spacebar_width_multiplier = 6
item_height = 80
list_font_size = 48  # Larger font size for list items
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
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))  # Scale it to your window size

overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
overlay.fill((50, 50, 50, 240))  # Grey color with 50% transparency

# Define areas
scroll_list_area = pygame.Rect(0, 0, *scroll_list_area_size)
current_text_area = pygame.Rect(0, scroll_list_area.bottom, *current_text_area_size)
keyboard_layout_area = pygame.Rect(0, current_text_area.bottom + 50, *keyboard_layout_area_size)

# Keyboard layout setup
keyboard_rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM9", " "]
first_key_x = keyboard_layout_area.x + (keyboard_layout_area.width - (len(keyboard_rows[0]) * (button_size[0] + 5))) // 2
first_key_y = keyboard_layout_area.y + 50


# Scrollable list and rotatable square setup
list_items = ['Item {}'.format(i) for i in range(1, 21)]
scroll_y = 0
list_font = pygame.font.SysFont(None, list_font_size)
rotatable_square = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
rotatable_square.fill(CYAN)
square_center = (50 + square_size // 2, current_text_area.bottom + square_size // 2 + 50)
square_angle = 0

# Dragging states
list_dragging = False
rotating = False
drag_start_y = 0

# Initialize variables for text input and cursor
input_text = ''
cursor = '|'
cursor_visible = True  # Cursor visibility state
cursor_switch_ms = 500  # Cursor visibility switch time in milliseconds
last_switch = pygame.time.get_ticks()

last_angle = 0
rotating = False

list_items = ["Variable1", "Variable2", "Variable3", "Variable4"]

first_key_x = keyboard_layout_area.x + 50  # Adjust this as needed for padding from the left edge
first_key_y = keyboard_layout_area.y + 50

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



def handle_key_presses(event):
    global input_text, list_items
    if event.type == pygame.MOUSEBUTTONDOWN:
        for i, row in enumerate(keyboard_rows):
            for j, key in enumerate(row):
                key_x = first_key_x + keyboard_row_offsets[min(i, len(keyboard_row_offsets) - 1)] + j * (button_size[0] + 5)
                key_y = first_key_y + i * (button_size[1] + 5)
                key_width = button_size[0]
                if key == " ":
                    key_rect.width = button_size[0] * spacebar_width_multiplier
                key_rect = pygame.Rect(key_x, key_y, key_width, button_size[1])
                if key_rect.collidepoint(event.pos):
                    if key == "9":
                        # When Enter is pressed, add input_text to the list
                        if input_text.strip():  # Only add non-empty text
                            list_items.append(input_text)
                            input_text = ''  # Clear the input text
                    else:
                        input_text += key  # Add the key to the input text

# Game loop
running = True
while running:
    screen.fill(BLACK)
    current_time = pygame.time.get_ticks()

    # Toggle cursor visibility
    if current_time - last_switch > cursor_switch_ms:
        cursor_visible = not cursor_visible
        last_switch = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if scroll_list_area.collidepoint(event.pos):
                list_dragging = True
                drag_start_y = event.pos[1] - scroll_y  # Store the start position of the drag
            elif rotatable_square.get_rect(topleft=(square_center[0] - square_size // 2, square_center[1] - square_size // 2)).collidepoint(event.pos):
                rotating = True
                offset_x = event.pos[0] - square_center[0]
                offset_y = event.pos[1] - square_center[1]
                last_angle = math.atan2(offset_y, offset_x)
        elif event.type == pygame.MOUSEBUTTONUP:
            list_dragging = False
            rotating = False
        elif event.type == pygame.MOUSEMOTION:
            if list_dragging:
                # Calculate new scroll position based on mouse movement
                mouse_y = event.pos[1]
                scroll_y = mouse_y - drag_start_y
                # Clamp the scroll_y to prevent scrolling beyond the content
                max_scroll_y = -(item_height * len(list_items) - scroll_list_area.height)
                scroll_y = max(min(scroll_y, 0), max_scroll_y)
            elif rotating:
                # Calculate rotation based on mouse movement
                mouse_x, mouse_y = event.pos
                offset_x = mouse_x - square_center[0]
                offset_y = mouse_y - square_center[1]
                current_angle = math.atan2(offset_y, offset_x)
                angle_delta = math.degrees(current_angle - last_angle)
                last_angle = current_angle
                square_angle += angle_delta
                square_angle %= 360  # Keep the angle within 0-359 degrees

                # Adjust scroll position based on rotation
                total_scroll_height = item_height * len(list_items) - scroll_list_area.height
                scroll_per_degree = total_scroll_height / 360
                scroll_y = -(square_angle * scroll_per_degree) % total_scroll_height
                scroll_y = max(min(scroll_y, 0), -total_scroll_height)
        handle_key_presses(event)

        # Virtual key press handling
        

    screen.blit(background_image, (0, 0))
    screen.blit(overlay, (0, 0))
    # Draw elements
    draw_keyboard()
    draw_scroll_list()

    # Draw current text area with input text and cursor
    pygame.draw.rect(screen, GREY_BACKGROUND, current_text_area)
    text_surface = font.render(input_text + (cursor if cursor_visible else ''), True, BLUE_TRON)
    screen.blit(text_surface, (current_text_area.x+10, current_text_area.y+10))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
