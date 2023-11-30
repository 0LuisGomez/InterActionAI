import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Define colors and constants
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PURPLE = (150, 0, 150)
BLACK = (0, 0, 0)

# Window size
WINDOW_WIDTH, WINDOW_HEIGHT = 1154, 655

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Interface Example")

# Define text areas, keyboard layout rect, and scrollable list area
previous_text_area = pygame.Rect(0, 0, WINDOW_WIDTH, 100)
current_text_area = pygame.Rect(0, previous_text_area.bottom, WINDOW_WIDTH, 50)
keyboard_layout_area = pygame.Rect(0, current_text_area.bottom + 50, WINDOW_WIDTH - 150, 300)
scroll_list_area = pygame.Rect(keyboard_layout_area.right, current_text_area.bottom + 50, 150, 300)

# Scrollable list setup
list_items = ['Item {}'.format(i) for i in range(1, 21)]
item_height = 30
scroll_y = 0
font = pygame.font.SysFont(None, 24)

# Rotatable square setup
square_size = 100
rotatable_square = pygame.Surface((square_size, square_size))
rotatable_square.fill(RED)
square_center = (50 + square_size // 2, WINDOW_HEIGHT - square_size // 2)
square_angle = 0

# Dragging states
list_dragging = False
rotating = False
last_mouse_y = 0
last_angle = 0

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if scroll_list_area.collidepoint(event.pos):
                list_dragging = True
                last_mouse_y = event.pos[1]
            elif rotatable_square.get_rect(topleft=(50, WINDOW_HEIGHT - square_size)).collidepoint(event.pos):
                rotating = True
                last_angle = math.atan2(square_center[1] - event.pos[1], event.pos[0] - square_center[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            list_dragging = False
            rotating = False
        elif event.type == pygame.MOUSEMOTION:
            if list_dragging:
                current_mouse_y = event.pos[1]
                delta_y = current_mouse_y - last_mouse_y
                last_mouse_y = current_mouse_y
                # Update scroll_y but prevent it from going out of bounds
                max_scroll_y = -(item_height * len(list_items) - scroll_list_area.height)
                scroll_y = max(min(scroll_y + delta_y, 0), max_scroll_y)
            elif rotating:
                current_angle = math.atan2(square_center[1] - event.pos[1], event.pos[0] - square_center[0])
                angle_delta = math.degrees(current_angle - last_angle)
                last_angle = current_angle
                square_angle += angle_delta
                # Normalize the angle between 0 and 360
                square_angle = square_angle % 360

    # Fill the background
    screen.fill(WHITE)

    # Draw text areas and keyboard layout
    pygame.draw.rect(screen, YELLOW, previous_text_area)
    pygame.draw.rect(screen, PURPLE, current_text_area)
    pygame.draw.rect(screen, RED, keyboard_layout_area, 2)  # Outline

    # Draw the scrollable list
    pygame.draw.rect(screen, BLACK, scroll_list_area, 2)  # Outline for the list area
    for index, item_text in enumerate(list_items):
        item_y = scroll_list_area.y + index * item_height + scroll_y
        if scroll_list_area.y < item_y + item_height and item_y < scroll_list_area.y + scroll_list_area.height:
            item_surface = font.render(item_text, True, BLACK)
            screen.blit(item_surface, (scroll_list_area.x + 5, item_y))

    # Draw and rotate the square
    rotated_square = pygame.transform.rotate(rotatable_square, square_angle)
    rotated_rect = rotated_square.get_rect(center=square_center)
    screen.blit(rotated_square, rotated_rect.topleft)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
