import logging
import os
from cmath import inf
import pygame
import random
import time
import explorepy
import argparse
import csv
from MockExplore import MockExplore
from Character import Character
from train_model import predict_for_pygame, setup_for_training, train_and_predict
from analysis import print_predict_and_truth

# Disable explorepy custom excepthook
import sys
sys.excepthook = sys.__excepthook__

# SETUP GLOBAL CONSTANT
# Time in s between each row / column
STIMULUS_INTERVAL = 1 / 16
# Time that a row is intensified for
INTENSIFICATION_DURATION = STIMULUS_INTERVAL * 0.5
# Number of times each row and column will be cycled through before a break
N_CYCLES_IN_EPOCH = 5  # Also know as N_REPEAT
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


# WARNING !!!!!!!!!
# WARNING: We used the int() type caster in the explore.setMarker. This can cause issues in the future if we use letters.
# Letter -> Mapping can be used in the future.
# WARNING !!!!!!!!!


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example code for marker generation")
    parser.add_argument("-n", "--name", dest="name", default="Explore_842F", type=str, help="Name of the device")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="data/default/default",
        type=str,
        help="Name of the output files",
    )
    parser.add_argument("--mock", dest="mock", help="Use a mock Mentalab Explore device", action="store_true")
    parser.add_argument("--model", dest="model", default="model.joblib", type=str, help="Specify the filename of trained model to load")
    parser.add_argument("--mode", dest="mode", default="predict", type=str, help="Run mode: train / predict / full")
    parser.add_argument("--train_seq", dest="train_seq", default="139726845", type=str, help="Sequence of number for training. Example: 139726845")

    args = parser.parse_args()
    return args


def create_explore_object(args, print_marker=False):
    args = parse_arguments()
    # Create an Explore object
    if args.mock:
        explore = MockExplore(log=False)
    else:
        explore = explorepy.Explore()
    explore.connect(device_name=args.name)
    # We don't need this data
    explore.disable_module('ORN')
    if not print_marker:
        # Disable logging
        logger = logging.getLogger('explorepy')
        logger.setLevel(level=logging.CRITICAL)
    explore.set_sampling_rate(250)
    return explore


def write_session_parameters(args):
    with open(args.output + "_Param.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stimulus_interval",
                "intensification_duration",
                "n_cycles_in_epoch",
                "auto_epoch",
                "break_time",
                "displayed_chars",
                "matrix_dimensions",
                "font_size",
                "flash_type",
                "screen_size",
            ]
        )
        writer.writerow(
            [
                STIMULUS_INTERVAL,
                INTENSIFICATION_DURATION,
                N_CYCLES_IN_EPOCH,
                AUTO_EPOCH,
                BREAK_TIME,
                DISPLAYED_CHARS,
                MATRIX_DIMENSIONS,
                FONT_SIZE,
                FLASH_TYPE,
                SCREEN_SIZE,
            ]
        )


def init_char_array(starting_x_pos, reserved_space, char_surface_size, explore):
    font = pygame.font.Font("HelveticaBold.ttf", FONT_SIZE)
    # Initialises all the characters to be shown on screen, saves each character in their group
    # each group is either a row or column
    chars = []
    # groups = [[] for i in range(MATRIX_DIMENSIONS[0] + MATRIX_DIMENSIONS[1])]
    i = 1
    row = 0
    col = 0
    pos = [starting_x_pos, reserved_space]
    for char in DISPLAYED_CHARS:
        chars.append(Character(char, tuple(pos), char_surface_size, font, FLASH_TYPE, explore))
        # groups[row].append(i - 1)
        # groups[col + MATRIX_DIMENSIONS[0]].append(i - 1)
        if i % MATRIX_DIMENSIONS[0] == 0 and i != 0:
            pos[0] = starting_x_pos
            pos[1] += char_surface_size[1]
            col = 0
            row += 1
        else:
            pos[0] += char_surface_size[0]
            col += 1
        i += 1
    random.shuffle(chars)
    return chars


def check_user_event(explore, epoch_on):
    """
    Checks user events to exit program or restart epoch.
    Press SPACE to continue.
    """
    for event in pygame.event.get():
        # exits program if user presses exit
        if event.type == pygame.QUIT:
            explore.stop_recording()
            pygame.quit()
            exit()
        # restarts epoch if user presses space
        if not AUTO_EPOCH and not epoch_on and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return True
    return False


def do_the_prediction_thingy(is_training, args, explore, screen, past_predictions):
    if (is_training):
        return
    # Try to record some extra data before stop
    time.sleep(0.8)
    explore.stop_recording()
    # if args.mock:
    #     n = len(DISPLAYED_CHARS)
    #     pred = DISPLAYED_CHARS[random.randint(0, n-1)]
    # else:
    pred = predict_for_pygame(args.output, args.model)
    pred = str(pred)
    print("Predicted: ", pred)
    if len(past_predictions) != 0:
        pos = (past_predictions[-1].screen_position[0] + 50, 25)
    else:
        pos = (50, 25)
    font = pygame.font.Font("HelveticaBold.ttf", 60)
    char = Character(pred, pos, (50, 50), font, FLASH_TYPE, explore)
    past_predictions.append(char)
    screen.blit(char.surface, char.screen_position)


def main_train(num_markers, filename):
    data, y, info, y_val, num_markers = setup_for_training(filename, 5, num_markers)
    y, preds_nonUS = train_and_predict(data, y, info)
    print_predict_and_truth(y, y_val, preds_nonUS, num_markers)


def main_speller(explore, args, is_training=False, n_epochs=-1):
    # Reserve some space on the top for predicted characters
    reserved_space = 100

    # finds derived values
    starting_x_pos = 0
    char_surface_size = (SCREEN_SIZE[0] / MATRIX_DIMENSIONS[0], (SCREEN_SIZE[1] - reserved_space) / MATRIX_DIMENSIONS[1])
    if SQUARE_SHAPE_DISTRIBUTION:
        tmp = min(char_surface_size[0], char_surface_size[1])
        char_surface_size = (tmp, tmp)
        starting_x_pos = (SCREEN_SIZE[0] - tmp * MATRIX_DIMENSIONS[0]) / 2

    # Initialises the pygame screen
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    if is_training:
        pygame.display.set_caption('Calibrating...')
    else:
        pygame.display.set_caption('Live spelling...')
    screen.fill('gray')
    grid_surface = pygame.Surface((SCREEN_SIZE[0], SCREEN_SIZE[1] - reserved_space))
    grid_surface.fill('black')
    screen.blit(grid_surface, (0, reserved_space))
    clock = pygame.time.Clock()
    chars = init_char_array(starting_x_pos, reserved_space, char_surface_size, explore)

    # starts game loop
    group_num = 0
    time_since_intensification = time.time()
    row_intensified = False
    epoch_on = True
    n_cycles = 0
    time_end_epoch = inf
    pressed_spacebar = False
    explore.record_data(file_name=args.output, file_type="csv", do_overwrite=True, block=False)
    epoch_counter = 0
    predictedCharacters = []

    # If not in training mode, run forever
    # Else, run in n_epochs loops.
    while (epoch_counter < n_epochs or not is_training):
        # defines the time of the frame so that function does not need to be called often
        time_of_frame = time.time()

        # checks user events to exit program or restart epoch by pressing spacebar
        pressed_spacebar = check_user_event(explore, epoch_on)

        # restarts epoch if pressed spacebar OR on automatic and enough time has passed
        if not epoch_on:
            if pressed_spacebar or (AUTO_EPOCH and time_of_frame - time_end_epoch > BREAK_TIME):
                epoch_on = True
                n_cycles = 0
                epoch_counter += 1
                pressed_spacebar = False
                if not is_training:
                    explore.record_data(file_name=args.output, file_type="csv", do_overwrite=True, block=False)

        if epoch_on:
            # intensifies new group of chars and puts them on the screen
            # FIXHERE
            if time_of_frame - time_since_intensification > STIMULUS_INTERVAL:
                # shuffles groups and starts again if all groups have been intensified
                if group_num == MATRIX_DIMENSIONS[0] * MATRIX_DIMENSIONS[1]:
                    # ensures that the intensified row / col is not flashed consecutively
                    shuffled = False
                    last_char = chars[group_num - 1]
                    while not shuffled:
                        random.shuffle(chars)
                        if last_char != chars[0]:
                            shuffled = True
                    group_num = 0
                    n_cycles += 1
                else:
                    # intensifies new char
                    chars[group_num].intensify()
                    group_num += 1

                # refreshes and puts all characters on screen
                grid_surface.fill("black")
                for char in chars:
                    screen.blit(char.surface, char.screen_position)
                row_intensified = True
                time_since_intensification = time.time()
                # ends epoch if completed n_cycles in epoch
                if n_cycles >= N_CYCLES_IN_EPOCH:
                    epoch_on = False
                    time_end_epoch = time.time()
                    print("END EPOCHS", epoch_counter)
                    do_the_prediction_thingy(is_training, args, explore, screen, predictedCharacters)

        # darkens rows / cols after they have been on longer then intensification duration
        if row_intensified and time_of_frame - time_since_intensification > INTENSIFICATION_DURATION:
            # darkens char
            chars[group_num - 1].darken()
            # refreshes and puts all characters on screen
            grid_surface.fill("black")
            for char in chars:
                screen.blit(char.surface, char.screen_position)
            row_intensified = False

        pygame.display.update()
        clock.tick(FPS)
    time.sleep(0.8)
    explore.stop_recording()
    pygame.quit()


def main():
    # Get config input
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    explore = create_explore_object(args)
    write_session_parameters(args)

    # Do the calibrating + training
    if (args.mode == "train" or args.mode == "full"):
        num_markers = [int(item) for item in list(args.train_seq)]
        print("=====================TRAIN=====================")
        main_speller(explore, args, True, len(num_markers))
        main_train(num_markers, args.output)

    # Do the spelling
    print("=====================LIVE SPELLING=====================")
    # Should run forever until force termination
    main_speller(explore, args)
    return


if __name__ == "__main__":
    main()
