"""Helpers for condition-heavy transformation protocol steps."""

import time
from aira.robot import arm


def grab_pipette(args):
    right = arm('right')
    which_one = args.get('which_one', 'left')
    let_go = bool(args.get('let_go', False))
    right.go_to('coop_right_home')
    right.go_to('coop_right_pipette_prep')
    right.set_gripper_position(600)
    if which_one == 'right':
        right.go_to('coop_right_pipette_right')
        right.go_to('coop_right_pipette_right_2')
        right.go_to('coop_right_pipette_right_3', offset=[0, 0, 35])
    elif which_one == 'middle':
        right.go_to('coop_right_pipette_middle', offset=[0, 0, 30])
    elif which_one == 'left':
        right.go_to('coop_right_pipette_left')
        right.go_to('coop_right_pipette_left_2', offset=[0, 0, 30])
    else:
        raise ValueError(f'Invalid pipette position: {which_one}')
    right.set_gripper_position(130)
    right.base_move(-5, 0, 8, speed=150, acc=200)
    right.base_move(-70, 0, 15, speed=200, acc=200)
    right.go_to('coop_right_pipette_prep')
    right.go_to('coop_right_pippete_give')
    if let_go:
        time.sleep(1.5)
        right.set_gripper_position(400)
        right.go_to('coop_right_home')


def place_pipette(args):
    right = arm('right')
    which_one = args.get('which_one', 'left')
    right.go_to('coop_right_pipette_prep')
    offset = [-20, 0, 50]
    if which_one == 'right':
        right.go_to('coop_right_pipette_right')
        right.go_to('coop_right_pipette_right_2')
        right.go_to('coop_right_pipette_right_3', offset=offset)
        right.go_to('coop_right_pipette_right_3')
    elif which_one == 'middle':
        right.go_to('coop_right_pipette_middle', offset=offset)
        right.go_to('coop_right_pipette_middle_2', offset=[0, 0, 0])
        right.go_to('coop_right_pipette_middle_3', offset=[0, 0, 0])
        right.go_to('coop_right_pipette_middle_4', offset=[0, 0, 0])
    elif which_one == 'left':
        right.go_to('trans_place_first_pipette')
        right.go_to('trans_place_first_pipette_2')
        right.go_to('trans_place_first_pipette_3')
    else:
        raise ValueError(f'Invalid pipette position: {which_one}')
    right.set_gripper_position(400)
    right.base_move(-40, 0, 5, speed=200, acc=200)
    right.go_to('coop_right_pipette_prep')
    right.go_to('coop_right_home')
