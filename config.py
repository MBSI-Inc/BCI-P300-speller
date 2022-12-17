# SETUP GLOBAL CONSTANT
# Time in s between each row / column
STIMULUS_INTERVAL = 1 / 16
# Time that a row is intensified for
INTENSIFICATION_DURATION = STIMULUS_INTERVAL * 0.5
# Number of times each row and column will be cycled through before a break
N_CYCLES_IN_EPOCH = 5
# Determines if the break is automatic or epoch is reinitiated by user pressing space
AUTO_EPOCH = True
# Break time in seconds between epochs
BREAK_TIME = 1
DISPLAYED_CHARS = "123456789".upper()
MATRIX_DIMENSIONS = (3, 3)
FONT_SIZE = 120
# Choose type of flash highlight
# 0 = Character only
# 1 = Surface area
# 2 = Character box area (small area around character)
FLASH_TYPE = 2
# POssible screen size
SCREEN_SIZE_SETTINGS = [(800, 600), (1280, 720), (1600, 900)]
SCREEN_SIZE = SCREEN_SIZE_SETTINGS[0]
# Whether the distribution of character is fully spread out in rectangle shape or
# focus in square-ish shape
SQUARE_SHAPE_DISTRIBUTION = False
FPS = 60

N_CHAR = 9
N_FLASH = 5
N_FLASH_PER_CHAR = 45
N_SPELLED = 27  # How many character I spelled
