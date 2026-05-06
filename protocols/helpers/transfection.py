"""Fallback helpers for the transfection YAML protocol."""

import math
import random
import threading
import time

from aira.robot import arm


PIPETTE_1000_DISH_TOP_LEVEL = 252.7
PIPETTE_1000_DISH_SECOND_STOP = 249.5


def pipette_to_dish(args):
    """Imperative fallback for the legacy transfection dish dispense loop."""
    right = arm('right')
    left = arm('left')
    n_steps_per_dish = int(args.get('n_steps_per_dish', 9))
    radius_mm = float(args.get('radius_mm', 30.0))
    n_dishes = int(args.get('n_dishes', 3))
    dish_spacing_mm = float(args.get('dish_spacing_mm', 90.0))
    speed = float(args.get('speed', 80))
    acc = float(args.get('acc', 150))

    top = float(args.get('z_top', PIPETTE_1000_DISH_TOP_LEVEL))
    bottom = float(args.get('z_bottom', PIPETTE_1000_DISH_SECOND_STOP))
    step_depth = (top - bottom) / n_steps_per_dish
    left.set_gripper_position(0)

    for d in range(n_dishes):
        right.go_to('tf_dish_1', offset=[0, -d * dish_spacing_mm, 0])
        left.go_to('tf_dish_2', offset=[0, -d * dish_spacing_mm, 0])
        prev_x, prev_y = 0.0, 0.0
        for i in range(n_steps_per_dish + 1):
            left.z_level(top - step_depth * i, speed=speed, acc=acc)
            if i == n_steps_per_dish:
                break
            slice_lo = 2 * math.pi * i / n_steps_per_dish
            slice_hi = 2 * math.pi * (i + 1) / n_steps_per_dish
            r = radius_mm * math.sqrt(random.random())
            theta = random.uniform(slice_lo, slice_hi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            dx, dy = x - prev_x, y - prev_y
            tr = threading.Thread(target=right.base_move, args=(dx, dy, 0), kwargs={'speed': speed, 'acc': acc})
            tl = threading.Thread(target=left.base_move, args=(dx, dy, 0), kwargs={'speed': speed, 'acc': acc})
            tr.start(); tl.start(); tr.join(); tl.join()
            prev_x, prev_y = x, y
        left.base_move(0, 0, 30, speed=speed, acc=acc)
        left.go_to('tf_left_home')
        right.z_level(400)
        if d < n_dishes - 1:
            right.go_to('tf_dish_pip_1')
            right.go_to('tf_dish_pip_2')
