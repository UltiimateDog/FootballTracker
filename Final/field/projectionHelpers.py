import numpy as np
import cv2

from Final.field.KeyPoints import back_middle_line_world, front_middle_line_world, corner_back_left_world, \
    corner_front_left_world, corner_front_right_world, corner_back_right_world
from Final.Constants import (
    LINE_COLOR, LINE_THICKNESS, CIRCLE_RADIUS, CIRCLE_RESOLUTION,
    PENALTY_LEFT_FRONT_GOAL_WORLD, PENALTY_LEFT_FRONT_FIELD_WORLD,
    PENALTY_LEFT_BACK_FIELD_WORLD, PENALTY_LEFT_BACK_GOAL_WORLD,
    PENALTY_RIGHT_FRONT_GOAL_WORLD, PENALTY_RIGHT_FRONT_FIELD_WORLD,
    PENALTY_RIGHT_BACK_FIELD_WORLD, PENALTY_RIGHT_BACK_GOAL_WORLD
)


def project_to_screen(K, to_device_from_world, point_in_world):
    """
    Project point_in_world to the screen and returns pixel coordinates
    """
    homog = np.ones((4, 1))
    homog[0:3, 0] = point_in_world
    point_in_device = np.dot(to_device_from_world, homog)
    point_in_device_divided_depth = (point_in_device / point_in_device[2, 0])[0:3]
    point_projected = np.dot(K, point_in_device_divided_depth)
    return [int(point_projected[0]), int(point_projected[1])]


def project_and_draw_lines(K, to_device_from_world, points_in_world, img):
    """Project list of points in world and draw polyline"""

    projected_points = []
    for point_in_world in points_in_world:
        projected_points.append(
            project_to_screen(K, to_device_from_world, np.array(point_in_world))
        )

    nb_pts = len(projected_points)
    for i in range(nb_pts):
        img = cv2.line(
            img,
            projected_points[i],
            projected_points[(i + 1) % nb_pts],
            color=LINE_COLOR,
            thickness=LINE_THICKNESS,
        )

    return img


def draw_central_circle(K, to_device_from_world, img):
    """Draw central circle on the image"""

    res = CIRCLE_RESOLUTION
    circle_points_projected = np.zeros((res, 2), dtype=np.int32)
    for i in range(res):
        angle = i / res * np.pi * 2
        circle_points_world = (
            np.array([np.cos(angle), 0, np.sin(angle)]) * CIRCLE_RADIUS
        )
        circle_points_projected[i] = project_to_screen(
            K, to_device_from_world, circle_points_world
        )

    img = cv2.polylines(
        img, [circle_points_projected], isClosed=True, color=LINE_COLOR, thickness=LINE_THICKNESS
    )

    return img


def draw_middle_line(K, to_device_from_world, img):
    """Draw middle/main line"""

    img = project_and_draw_lines(
        K,
        to_device_from_world,
        [
            back_middle_line_world,
            front_middle_line_world,
        ],
        img,
    )

    return img


def draw_border_lines(K, to_device_from_world, img):
    """Draw border lines"""

    img = project_and_draw_lines(
        K,
        to_device_from_world,
        [
            corner_back_left_world,
            corner_front_left_world,
            corner_front_right_world,
            corner_back_right_world,
        ],
        img,
    )

    return img


def draw_penalty_areas(K, to_device_from_world, img):
    """
    Draw penalty areas on both sides
    """

    img = project_and_draw_lines(
        K,
        to_device_from_world,
        [
            PENALTY_LEFT_FRONT_GOAL_WORLD,
            PENALTY_LEFT_FRONT_FIELD_WORLD,
            PENALTY_LEFT_BACK_FIELD_WORLD,
            PENALTY_LEFT_BACK_GOAL_WORLD,
        ],
        img,
    )

    img = project_and_draw_lines(
        K,
        to_device_from_world,
        [
            PENALTY_RIGHT_FRONT_GOAL_WORLD,
            PENALTY_RIGHT_FRONT_FIELD_WORLD,
            PENALTY_RIGHT_BACK_FIELD_WORLD,
            PENALTY_RIGHT_BACK_GOAL_WORLD,
        ],
        img,
    )

    return img


def draw_pitch_lines(K, to_device_from_world, img):
    """Draw pitch lines"""
    img = draw_central_circle(K, to_device_from_world, img)
    img = draw_middle_line(K, to_device_from_world, img)
    img = draw_border_lines(K, to_device_from_world, img)
    img = draw_penalty_areas(K, to_device_from_world, img)

    return img
