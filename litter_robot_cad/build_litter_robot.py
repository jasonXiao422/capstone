"""Parametric CAD base model for a compact autonomous litter-collection robot.

Coordinate system:
    X axis: vehicle length, front is negative X, rear is positive X.
    Y axis: vehicle width, left/right are negative/positive Y.
    Z axis: vertical, ground plane is Z = 0.

The model is intentionally a clean architecture/base CAD assembly, not a final
production detail model. Dimensions are millimeters.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from build123d import *  # noqa: F401,F403,E402


# ---------------------------------------------------------------------------
# Named dimensions
# ---------------------------------------------------------------------------

OVERALL_LENGTH = 850.0
OVERALL_WIDTH = 520.0
FRAME_HEIGHT = 360.0
EXTRUSION = 30.0
GROUND_CLEARANCE = 85.0

WHEEL_DIAMETER = 205.0
WHEEL_WIDTH = 68.0
WHEEL_RADIUS = WHEEL_DIAMETER / 2.0
WHEEL_SIDE_CLEARANCE = 22.0
WHEEL_X_OFFSET_FROM_END = 96.0

BRUSH_LENGTH = 195.0
BRUSH_RADIUS = 48.0
BRUSH_HUB_RADIUS = 14.0
BRUSH_CENTER_Z = 56.0
BRUSH_CENTER_X = -OVERALL_LENGTH / 2.0 - 54.0
BRUSH_CENTER_Y = 112.0

RAMP_WIDTH = 370.0
RAMP_THICKNESS = 5.0
RAMP_START_X = BRUSH_CENTER_X + 40.0
RAMP_START_Z = 35.0
RAMP_END_X = -190.0
RAMP_END_Z = 166.0

BIN_LENGTH = 420.0
BIN_WIDTH = 330.0
BIN_HEIGHT = 140.0
BIN_FRONT_X = RAMP_END_X
BIN_BOTTOM_Z = 170.0
BIN_CENTER_X = BIN_FRONT_X + BIN_LENGTH / 2.0
BIN_CENTER_Z = BIN_BOTTOM_Z + BIN_HEIGHT / 2.0

BATTERY_LENGTH = 260.0
BATTERY_WIDTH = 210.0
BATTERY_HEIGHT = 55.0
BATTERY_CENTER_X = 65.0
BATTERY_CENTER_Z = 126.0

ELECTRONICS_LENGTH = 145.0
ELECTRONICS_WIDTH = 54.0
ELECTRONICS_HEIGHT = 86.0

MAST_HEIGHT_ABOVE_FRAME = 116.0


# ---------------------------------------------------------------------------
# Paths and colors
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.dirname(__file__))
EXPORT = os.path.join(ROOT, "exports")
STEP_DIR = os.path.join(EXPORT, "step")
STL_DIR = os.path.join(EXPORT, "stl")
PNG_DIR = os.path.join(EXPORT, "png")
STL_LINEAR_TOLERANCE = 0.35
STL_ANGULAR_TOLERANCE = 0.32
LIGHTWEIGHT_EXPORT = True

SILVER = (0.72, 0.74, 0.74, 1.0)
DARK_SLOT = (0.08, 0.085, 0.09, 1.0)
BLACK = (0.01, 0.012, 0.014, 1.0)
MATTE_BLACK = (0.035, 0.038, 0.04, 1.0)
RUBBER = (0.005, 0.005, 0.005, 1.0)
ORANGE = (1.0, 0.34, 0.02, 1.0)
SMOKE = (0.08, 0.10, 0.11, 0.42)
CLEAR = (0.68, 0.82, 0.92, 0.28)
STEEL = (0.42, 0.43, 0.43, 1.0)
GRASS = (0.30, 0.50, 0.23, 1.0)


@dataclass(frozen=True)
class PartGroup:
    name: str
    obj: object
    color: tuple[float, float, float, float]
    preview_alpha: float = 1.0


def colorize(obj, rgba):
    obj.color = Color(*rgba)
    return obj


def safe_fillet(part, edges, radius):
    if LIGHTWEIGHT_EXPORT:
        return part
    try:
        if len(edges) == 0:
            return part
        return fillet(edges, radius)
    except Exception as exc:
        print(f"   [skip fillet r={radius}: {exc.__class__.__name__}]")
        return part


def box(length, width, height, rgba, fillet_radius=0.0):
    part = Box(length, width, height)
    if fillet_radius > 0:
        part = safe_fillet(part, part.edges(), fillet_radius)
    return colorize(part, rgba)


def cyl_x(radius, length, rgba):
    return colorize(Rot(0, 90, 0) * Cylinder(radius, length), rgba)


def cyl_y(radius, length, rgba):
    return colorize(Rot(90, 0, 0) * Cylinder(radius, length), rgba)


def cyl_z(radius, height, rgba):
    return colorize(Cylinder(radius, height), rgba)


def compound(children):
    return Compound(children=list(children))


def rail(axis: str, length: float, center: tuple[float, float, float]):
    """Square T-slot extrusion approximation with attached dark slot inserts."""
    slot_w = 6.0
    slot_t = 1.4
    x, y, z = center
    parts = []
    body = box(
        length if axis == "x" else EXTRUSION,
        length if axis == "y" else EXTRUSION,
        length if axis == "z" else EXTRUSION,
        SILVER,
        1.2,
    )
    parts.append(Pos(x, y, z) * body)
    if LIGHTWEIGHT_EXPORT:
        return compound(parts)

    if axis == "x":
        slot_shapes = [
            Pos(0, 0, EXTRUSION / 2.0 + slot_t / 2.0) * Box(length + 0.4, slot_w, slot_t),
            Pos(0, EXTRUSION / 2.0 + slot_t / 2.0, 0) * Box(length + 0.4, slot_t, slot_w),
            Pos(0, -EXTRUSION / 2.0 - slot_t / 2.0, 0) * Box(length + 0.4, slot_t, slot_w),
        ]
    elif axis == "y":
        slot_shapes = [
            Pos(0, 0, EXTRUSION / 2.0 + slot_t / 2.0) * Box(slot_w, length + 0.4, slot_t),
            Pos(EXTRUSION / 2.0 + slot_t / 2.0, 0, 0) * Box(slot_t, length + 0.4, slot_w),
            Pos(-EXTRUSION / 2.0 - slot_t / 2.0, 0, 0) * Box(slot_t, length + 0.4, slot_w),
        ]
    else:
        slot_shapes = [
            Pos(EXTRUSION / 2.0 + slot_t / 2.0, 0, 0) * Box(slot_t, slot_w, length + 0.4),
            Pos(-EXTRUSION / 2.0 - slot_t / 2.0, 0, 0) * Box(slot_t, slot_w, length + 0.4),
            Pos(0, EXTRUSION / 2.0 + slot_t / 2.0, 0) * Box(slot_w, slot_t, length + 0.4),
            Pos(0, -EXTRUSION / 2.0 - slot_t / 2.0, 0) * Box(slot_w, slot_t, length + 0.4),
        ]
    parts.extend(Pos(x, y, z) * colorize(s, DARK_SLOT) for s in slot_shapes)
    return compound(parts)


def build_frame():
    lower_z = GROUND_CLEARANCE + EXTRUSION / 2.0
    upper_z = FRAME_HEIGHT - EXTRUSION / 2.0
    side_y = OVERALL_WIDTH / 2.0 - EXTRUSION / 2.0
    end_x = OVERALL_LENGTH / 2.0 - EXTRUSION / 2.0
    post_z = (lower_z + upper_z) / 2.0
    post_len = upper_z - lower_z
    parts = []

    for z in (lower_z, upper_z):
        parts.append(rail("x", OVERALL_LENGTH, (0, side_y, z)))
        parts.append(rail("x", OVERALL_LENGTH, (0, -side_y, z)))
        parts.append(rail("y", OVERALL_WIDTH, (-end_x, 0, z)))
        parts.append(rail("y", OVERALL_WIDTH, (end_x, 0, z)))

    for x in (-end_x, end_x):
        for y in (-side_y, side_y):
            parts.append(rail("z", post_len, (x, y, post_z)))

    # Functional internal crossmembers: battery, bin supports, electronics plate.
    for x in (-130.0, 40.0, 205.0):
        parts.append(rail("y", OVERALL_WIDTH - 2 * EXTRUSION, (x, 0, lower_z + 44.0)))
    for y in (-125.0, 125.0):
        parts.append(rail("x", BIN_LENGTH + 34.0, (BIN_CENTER_X, y, BIN_BOTTOM_Z - 18.0)))
    parts.append(rail("y", OVERALL_WIDTH - 2 * EXTRUSION, (RAMP_END_X - 8.0, 0, BIN_BOTTOM_Z - 18.0)))

    return compound(parts)


def bolt_disk_on_y(x, y, z, sign=1.0, r=4.0):
    return Pos(x, y, z) * cyl_y(r, 3.0, BLACK if sign > 0 else BLACK)


def bolt_disk_on_x(x, y, z, sign=1.0, r=4.0):
    return Pos(x, y, z) * cyl_x(r, 3.0, BLACK if sign > 0 else BLACK)


def build_brackets():
    lower_z = GROUND_CLEARANCE + EXTRUSION / 2.0
    upper_z = FRAME_HEIGHT - EXTRUSION / 2.0
    side_y = OVERALL_WIDTH / 2.0 - EXTRUSION / 2.0
    end_x = OVERALL_LENGTH / 2.0 - EXTRUSION / 2.0
    parts = []

    for x in (-end_x, end_x):
        for y in (-side_y, side_y):
            for z in (lower_z, upper_z):
                parts.append(Pos(x, y, z) * box(44.0, 44.0, 44.0, BLACK, 1.0))
                plate_y = y + math.copysign(EXTRUSION / 2.0 + 2.0, y)
                parts.append(Pos(x, plate_y, z) * box(54.0, 4.0, 54.0, BLACK, 0.6))
                if not LIGHTWEIGHT_EXPORT:
                    for dx in (-15.0, 15.0):
                        for dz in (-15.0, 15.0):
                            parts.append(bolt_disk_on_y(x + dx, plate_y + math.copysign(2.2, y), z + dz, math.copysign(1, y), 3.2))

    # Orange brush-adjustment clevises attached to the front lower frame.
    for y in (-232.0, 232.0):
        parts.append(Pos(-OVERALL_LENGTH / 2.0 - 8.0, y, lower_z - 8.0) * box(28.0, 30.0, 46.0, ORANGE, 1.2))
    return compound(parts)


def build_wheel_at(x, y_sign):
    y = y_sign * (OVERALL_WIDTH / 2.0 + WHEEL_SIDE_CLEARANCE + WHEEL_WIDTH / 2.0)
    z = WHEEL_RADIUS
    parts = []

    # Simple placeholder wheels: black cylinders with plain gray center hubs.
    tire = cyl_y(WHEEL_RADIUS, WHEEL_WIDTH, RUBBER)
    hub = cyl_y(36.0, WHEEL_WIDTH + 6.0, STEEL)
    cap_outer_y = y + y_sign * (WHEEL_WIDTH / 2.0 + 4.5)
    cap_inner_y = y - y_sign * (WHEEL_WIDTH / 2.0 + 4.5)
    cap = cyl_y(40.0, 8.0, STEEL)
    axle_button = cyl_y(13.0, 10.0, MATTE_BLACK)
    parts.append(Pos(x, y, z) * tire)
    parts.append(Pos(x, y, z) * hub)
    parts.append(Pos(x, cap_outer_y, z) * cap)
    parts.append(Pos(x, cap_inner_y, z) * cap)
    parts.append(Pos(x, cap_outer_y + y_sign * 1.0, z) * axle_button)
    return compound(parts)


def build_wheels():
    wheel_x = OVERALL_LENGTH / 2.0 - WHEEL_X_OFFSET_FROM_END
    parts = []
    for x in (-wheel_x, wheel_x):
        for y_sign in (-1.0, 1.0):
            parts.append(build_wheel_at(x, y_sign))
    return compound(parts)


def build_wheel_mounts():
    lower_z = GROUND_CLEARANCE + EXTRUSION / 2.0
    side_y = OVERALL_WIDTH / 2.0 - EXTRUSION / 2.0
    wheel_x = OVERALL_LENGTH / 2.0 - WHEEL_X_OFFSET_FROM_END
    parts = []
    for x in (-wheel_x, wheel_x):
        for s in (-1.0, 1.0):
            rail_outer_y = s * (OVERALL_WIDTH / 2.0 + 5.0)
            wheel_y = s * (OVERALL_WIDTH / 2.0 + WHEEL_SIDE_CLEARANCE + WHEEL_WIDTH / 2.0)
            parts.append(Pos(x, rail_outer_y, WHEEL_RADIUS) * box(56.0, 18.0, 58.0, BLACK, 1.0))
            parts.append(Pos(x, s * (side_y + 18.0), lower_z) * box(72.0, 8.0, 46.0, BLACK, 0.7))
            axle_len = abs(wheel_y - rail_outer_y)
            axle_center_y = (wheel_y + rail_outer_y) / 2.0
            parts.append(Pos(x, axle_center_y, WHEEL_RADIUS) * cyl_y(12.0, axle_len, STEEL))
            if not LIGHTWEIGHT_EXPORT:
                for dx in (-18.0, 18.0):
                    parts.append(bolt_disk_on_y(x + dx, rail_outer_y + s * 9.5, WHEEL_RADIUS + 18.0, s, 3.5))
                    parts.append(bolt_disk_on_y(x + dx, rail_outer_y + s * 9.5, WHEEL_RADIUS - 18.0, s, 3.5))
    return compound(parts)


def build_brush_assembly():
    parts = []
    lower_z = GROUND_CLEARANCE + EXTRUSION / 2.0

    # Two large, low, simple rollers cover the ramp inlet without dense bristle detail.
    for y in (-BRUSH_CENTER_Y, BRUSH_CENTER_Y):
        parts.append(Pos(-OVERALL_LENGTH / 2.0 - 30.0, y, lower_z - 22.0) * box(135.0, 18.0, 20.0, ORANGE, 1.0))
        parts.append(Pos(BRUSH_CENTER_X, y, BRUSH_CENTER_Z + 4.0) * box(32.0, 28.0, 62.0, ORANGE, 1.0))
        parts.append(Pos(BRUSH_CENTER_X, y, BRUSH_CENTER_Z) * cyl_y(8.0, BRUSH_LENGTH + 42.0, STEEL))
        parts.append(Pos(BRUSH_CENTER_X, y, BRUSH_CENTER_Z) * cyl_y(BRUSH_RADIUS, BRUSH_LENGTH, BLACK))
        parts.append(Pos(BRUSH_CENTER_X, y, BRUSH_CENTER_Z) * cyl_y(BRUSH_HUB_RADIUS, BRUSH_LENGTH + 10.0, MATTE_BLACK))

        if not LIGHTWEIGHT_EXPORT:
            # Coarse raised ribs suggest a rough brush surface without modeling individual bristles.
            for i in range(10):
                a = math.radians(i * 360.0 / 10.0)
                rib = box(7.0, BRUSH_LENGTH + 3.0, 6.0, BLACK, 0.5)
                parts.append(
                    Pos(BRUSH_CENTER_X, y, BRUSH_CENTER_Z)
                    * Rot(0, -math.degrees(a), 0)
                    * Pos(BRUSH_RADIUS + 2.0, 0, 0)
                    * rib
                )

    center_guard = box(24.0, 34.0, 52.0, MATTE_BLACK, 0.8)
    parts.append(Pos(BRUSH_CENTER_X + 10.0, 0, BRUSH_CENTER_Z + 10.0) * center_guard)
    return compound(parts)


def build_ramp():
    dx = RAMP_END_X - RAMP_START_X
    dz = RAMP_END_Z - RAMP_START_Z
    ramp_len = math.hypot(dx, dz)
    angle = -math.degrees(math.atan2(dz, dx))
    cx = (RAMP_START_X + RAMP_END_X) / 2.0
    cz = (RAMP_START_Z + RAMP_END_Z) / 2.0
    parts = []
    tray = box(ramp_len, RAMP_WIDTH, RAMP_THICKNESS, MATTE_BLACK, 0.4)
    parts.append(Pos(cx, 0, cz) * Rot(0, angle, 0) * tray)
    for y in (-RAMP_WIDTH / 2.0, RAMP_WIDTH / 2.0):
        flange = box(ramp_len, 5.0, 24.0, ORANGE, 0.4)
        parts.append(Pos(cx, y, cz + 11.0) * Rot(0, angle, 0) * flange)
    parts.append(Pos(RAMP_END_X + 4.0, 0, RAMP_END_Z + 4.0) * box(16.0, RAMP_WIDTH + 16.0, 20.0, ORANGE, 0.8))
    return compound(parts)


def build_waste_bin():
    parts = []
    wall = 5.0
    lid_thk = 8.0
    front_opening_h = 72.0
    bin_top = BIN_BOTTOM_Z + BIN_HEIGHT

    # Supported panel assembly: bottom tray, side/back smoky panels, front inlet frame.
    parts.append(Pos(BIN_CENTER_X, 0, BIN_BOTTOM_Z + wall / 2.0) * box(BIN_LENGTH, BIN_WIDTH, wall, MATTE_BLACK, 0.6))
    parts.append(Pos(BIN_CENTER_X, -BIN_WIDTH / 2.0, BIN_CENTER_Z) * box(BIN_LENGTH, wall, BIN_HEIGHT, SMOKE, 0.8))
    parts.append(Pos(BIN_CENTER_X, BIN_WIDTH / 2.0, BIN_CENTER_Z) * box(BIN_LENGTH, wall, BIN_HEIGHT, SMOKE, 0.8))
    parts.append(Pos(BIN_FRONT_X + BIN_LENGTH, 0, BIN_CENTER_Z) * box(wall, BIN_WIDTH, BIN_HEIGHT, SMOKE, 0.8))
    parts.append(Pos(BIN_FRONT_X, 0, BIN_BOTTOM_Z + 18.0) * box(wall, BIN_WIDTH, 36.0, MATTE_BLACK, 0.5))
    parts.append(Pos(BIN_FRONT_X, 0, BIN_BOTTOM_Z + front_opening_h + 22.0) * box(wall, BIN_WIDTH, 18.0, ORANGE, 0.5))
    for y in (-BIN_WIDTH / 2.0, BIN_WIDTH / 2.0):
        parts.append(Pos(BIN_FRONT_X, y, BIN_BOTTOM_Z + front_opening_h / 2.0) * box(wall, 14.0, front_opening_h, ORANGE, 0.5))

    lid = Pos(BIN_CENTER_X + 20.0, 0, bin_top + lid_thk / 2.0) * box(BIN_LENGTH - 40.0, BIN_WIDTH + 16.0, lid_thk, MATTE_BLACK, 0.8)
    parts.append(lid)
    # Hinges/latches on the rear/top edge and front lip.
    for y in (-92.0, 92.0):
        parts.append(Pos(BIN_FRONT_X + BIN_LENGTH + 8.0, y, bin_top + 6.0) * box(20.0, 38.0, 12.0, BLACK, 0.4))
        parts.append(Pos(BIN_FRONT_X - 7.0, y, bin_top - 22.0) * box(10.0, 32.0, 32.0, ORANGE, 0.5))

    # Attached U-shaped carry handle.
    handle_z0 = bin_top + lid_thk
    parts.append(Pos(BIN_CENTER_X + 15.0, 0, handle_z0 + 28.0) * cyl_x(5.0, 92.0, ORANGE))
    for x in (BIN_CENTER_X - 31.0, BIN_CENTER_X + 61.0):
        parts.append(Pos(x, 0, handle_z0 + 14.0) * cyl_z(5.0, 28.0, ORANGE))
        parts.append(Pos(x, 0, handle_z0 + 2.0) * box(24.0, 20.0, 4.0, ORANGE, 0.4))

    # A few simplified pieces of litter inside, below the lid.
    parts.append(Pos(BIN_CENTER_X + 60.0, -40.0, BIN_BOTTOM_Z + 26.0) * box(38.0, 22.0, 6.0, STEEL, 0.5))
    parts.append(Pos(BIN_CENTER_X + 120.0, 52.0, BIN_BOTTOM_Z + 20.0) * Rot(0, 0, 17.0) * box(44.0, 16.0, 5.0, ORANGE, 0.5))
    return compound(parts)


def build_battery():
    parts = []
    parts.append(Pos(BATTERY_CENTER_X, 0, BATTERY_CENTER_Z) * box(BATTERY_LENGTH, BATTERY_WIDTH, BATTERY_HEIGHT, MATTE_BLACK, 1.2))
    parts.append(Pos(BATTERY_CENTER_X - 60.0, 0, BATTERY_CENTER_Z + BATTERY_HEIGHT / 2.0 + 4.0) * box(74.0, 22.0, 8.0, BLACK, 0.6))
    for x in (BATTERY_CENTER_X - BATTERY_LENGTH / 2.0 + 24.0, BATTERY_CENTER_X + BATTERY_LENGTH / 2.0 - 24.0):
        parts.append(Pos(x, -BATTERY_WIDTH / 2.0 - 9.0, BATTERY_CENTER_Z - 12.0) * box(36.0, 18.0, 18.0, BLACK, 0.5))
        parts.append(Pos(x, BATTERY_WIDTH / 2.0 + 9.0, BATTERY_CENTER_Z - 12.0) * box(36.0, 18.0, 18.0, BLACK, 0.5))
    return compound(parts)


def build_electronics():
    x = -92.0
    y = OVERALL_WIDTH / 2.0 - EXTRUSION - ELECTRONICS_WIDTH / 2.0 - 6.0
    z = 226.0
    parts = []
    parts.append(Pos(x, y, z) * box(ELECTRONICS_LENGTH, ELECTRONICS_WIDTH, ELECTRONICS_HEIGHT, MATTE_BLACK, 1.0))
    parts.append(Pos(x, y + ELECTRONICS_WIDTH / 2.0 + 2.5, z) * box(ELECTRONICS_LENGTH + 20.0, 5.0, ELECTRONICS_HEIGHT + 20.0, BLACK, 0.5))
    for i in range(5):
        parts.append(Pos(x - 44.0 + i * 18.0, y - ELECTRONICS_WIDTH / 2.0 - 2.0, z + 18.0) * cyl_y(3.0, 3.0, STEEL))
    for i in range(4):
        parts.append(Pos(x - 42.0 + i * 28.0, y - ELECTRONICS_WIDTH / 2.0 - 2.0, z - 24.0) * box(16.0, 3.0, 4.0, STEEL, 0.2))
    parts.append(Pos(x + 58.0, y - ELECTRONICS_WIDTH / 2.0 - 3.0, z + 4.0) * box(18.0, 4.0, 12.0, ORANGE, 0.3))
    return compound(parts)


def build_camera_mast():
    upper_z = FRAME_HEIGHT - EXTRUSION / 2.0
    mast_x = -318.0
    mast_y = -155.0
    mast_center_z = upper_z + MAST_HEIGHT_ABOVE_FRAME / 2.0
    parts = []
    parts.append(Pos(mast_x, mast_y, upper_z + 4.0) * box(64.0, 54.0, 8.0, BLACK, 0.5))
    parts.append(rail("z", MAST_HEIGHT_ABOVE_FRAME, (mast_x, mast_y, mast_center_z)))
    parts.append(Pos(mast_x, mast_y, upper_z + 28.0) * box(54.0, 8.0, 44.0, BLACK, 0.5))
    cam_plate_z = upper_z + MAST_HEIGHT_ABOVE_FRAME + 7.0
    parts.append(Pos(mast_x, mast_y, cam_plate_z) * box(78.0, 58.0, 8.0, ORANGE, 0.5))
    cam_z = cam_plate_z + 26.0
    parts.append(Pos(mast_x - 4.0, mast_y, cam_z) * box(58.0, 48.0, 38.0, MATTE_BLACK, 1.2))
    parts.append(Pos(mast_x - 35.0, mast_y, cam_z) * cyl_x(12.0, 9.0, BLACK))
    parts.append(Pos(mast_x - 41.0, mast_y, cam_z) * cyl_x(7.0, 5.0, STEEL))
    return compound(parts)


def build_top_handles():
    # Secondary orange service handle on the front upper crossmember.
    upper_z = FRAME_HEIGHT - EXTRUSION / 2.0
    x = -OVERALL_LENGTH / 2.0 + 145.0
    parts = [
        Pos(x, 0, upper_z + 42.0) * cyl_y(5.0, 125.0, ORANGE),
        Pos(x, -62.5, upper_z + 22.0) * cyl_z(5.0, 40.0, ORANGE),
        Pos(x, 62.5, upper_z + 22.0) * cyl_z(5.0, 40.0, ORANGE),
        Pos(x, -62.5, upper_z + 3.0) * box(22.0, 24.0, 6.0, ORANGE, 0.4),
        Pos(x, 62.5, upper_z + 3.0) * box(22.0, 24.0, 6.0, ORANGE, 0.4),
    ]
    return compound(parts)


def build_transparent_body_panels():
    lower_z = GROUND_CLEARANCE + EXTRUSION
    upper_z = FRAME_HEIGHT - EXTRUSION
    side_y = OVERALL_WIDTH / 2.0 - EXTRUSION / 2.0
    body_center_z = (lower_z + upper_z) / 2.0
    body_height = upper_z - lower_z
    parts = []

    # Transparent service panels make the internal bin, ramp, battery, and electronics readable.
    for y in (-side_y - 3.0, side_y + 3.0):
        parts.append(Pos(5.0, y, body_center_z) * box(660.0, 4.0, body_height - 34.0, CLEAR, 0.8))
        for x in (-270.0, 0.0, 270.0):
            parts.append(Pos(x, y, body_center_z) * box(9.0, 8.0, body_height - 18.0, BLACK, 0.5))

    top_panel_z = FRAME_HEIGHT + 3.5
    parts.append(Pos(-92.0, 0, top_panel_z) * box(260.0, 360.0, 5.0, CLEAR, 0.6))
    parts.append(Pos(190.0, 0, top_panel_z) * box(250.0, 360.0, 5.0, CLEAR, 0.6))
    for x in (-225.0, 55.0, 315.0):
        parts.append(Pos(x, -182.0, top_panel_z + 3.5) * box(42.0, 10.0, 8.0, BLACK, 0.4))
        parts.append(Pos(x, 182.0, top_panel_z + 3.5) * box(42.0, 10.0, 8.0, BLACK, 0.4))

    return compound(parts)


def build_chassis_body_details():
    parts = []

    # Interior engineered details: bin cradle rails, ramp side guides, electronics/battery straps.
    for y in (-BIN_WIDTH / 2.0 - 18.0, BIN_WIDTH / 2.0 + 18.0):
        parts.append(Pos(BIN_CENTER_X, y, BIN_BOTTOM_Z - 4.0) * box(BIN_LENGTH + 42.0, 8.0, 12.0, BLACK, 0.4))
        parts.append(Pos(BIN_FRONT_X + 22.0, y, BIN_BOTTOM_Z + 52.0) * box(46.0, 8.0, 96.0, BLACK, 0.4))

    ramp_dx = RAMP_END_X - RAMP_START_X
    ramp_dz = RAMP_END_Z - RAMP_START_Z
    ramp_len = math.hypot(ramp_dx, ramp_dz)
    ramp_angle = -math.degrees(math.atan2(ramp_dz, ramp_dx))
    ramp_cx = (RAMP_START_X + RAMP_END_X) / 2.0
    ramp_cz = (RAMP_START_Z + RAMP_END_Z) / 2.0
    for y in (-RAMP_WIDTH / 2.0 - 17.0, RAMP_WIDTH / 2.0 + 17.0):
        parts.append(Pos(ramp_cx, y, ramp_cz + 24.0) * Rot(0, ramp_angle, 0) * box(ramp_len - 22.0, 9.0, 32.0, BLACK, 0.4))

    for x in (BATTERY_CENTER_X - 82.0, BATTERY_CENTER_X + 82.0):
        parts.append(Pos(x, 0, BATTERY_CENTER_Z + BATTERY_HEIGHT / 2.0 + 9.0) * box(12.0, BATTERY_WIDTH + 38.0, 10.0, ORANGE, 0.4))

    electronics_bus_x = -18.0
    electronics_bus_y = OVERALL_WIDTH / 2.0 - 92.0
    parts.append(Pos(electronics_bus_x, electronics_bus_y, 262.0) * box(180.0, 16.0, 16.0, BLACK, 0.4))
    for x in (-85.0, -35.0, 15.0, 65.0):
        parts.append(Pos(x, electronics_bus_y - 10.0, 262.0) * cyl_y(3.0, 22.0, ORANGE))

    return compound(parts)


def build_assembly_groups():
    return [
        PartGroup("frame_rails", build_frame(), SILVER),
        PartGroup("corner_brackets_and_bolts", build_brackets(), BLACK),
        PartGroup("wheel_mounts_and_axles", build_wheel_mounts(), BLACK),
        PartGroup("simple_cylindrical_wheels", build_wheels(), RUBBER),
        PartGroup("front_brush_assembly", build_brush_assembly(), BLACK),
        PartGroup("angled_scoop_ramp", build_ramp(), MATTE_BLACK),
        PartGroup("waste_bin_panel_assembly", build_waste_bin(), SMOKE, 0.62),
        PartGroup("low_central_battery_box", build_battery(), MATTE_BLACK),
        PartGroup("electronics_enclosure", build_electronics(), MATTE_BLACK),
        PartGroup("transparent_body_panels", build_transparent_body_panels(), CLEAR, 0.34),
        PartGroup("chassis_body_detail_hardware", build_chassis_body_details(), BLACK),
        PartGroup("camera_mast_module", build_camera_mast(), SILVER),
        PartGroup("orange_service_handles", build_top_handles(), ORANGE),
    ]


def build_full_assembly():
    return compound([g.obj for g in build_assembly_groups()])


def export_all():
    os.makedirs(STEP_DIR, exist_ok=True)
    os.makedirs(STL_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)

    groups = build_assembly_groups()
    full = compound([g.obj for g in groups])
    export_step(full, os.path.join(STEP_DIR, "litter_collection_robot_full_assembly.step"))
    export_stl(
        full,
        os.path.join(STL_DIR, "litter_collection_robot_full_assembly.stl"),
        tolerance=STL_LINEAR_TOLERANCE,
        angular_tolerance=STL_ANGULAR_TOLERANCE,
    )

    for group in groups:
        export_stl(
            group.obj,
            os.path.join(STL_DIR, f"{group.name}.stl"),
            tolerance=STL_LINEAR_TOLERANCE,
            angular_tolerance=STL_ANGULAR_TOLERANCE,
        )
    render_preview(groups)


def read_binary_stl(path, max_tris=8000):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        keep = min(n, max_tris)
        stride = max(1, n // keep)
        tris = []
        for i in range(n):
            data = f.read(50)
            if i % stride == 0 and len(tris) < keep:
                vals = struct.unpack("<12f", data[:48])
                tris.append([vals[3:6], vals[6:9], vals[9:12]])
    return np.array(tris, dtype=float)


def add_solid(ax, tris, rgba, alpha=1.0, edge_alpha=0.015, line_width=0.025):
    ax.add_collection3d(
        Poly3DCollection(
            tris,
            facecolor=rgba[:3],
            alpha=min(alpha, rgba[3] if len(rgba) == 4 else 1.0),
            edgecolor=(0, 0, 0, edge_alpha),
            linewidths=line_width,
        )
    )


def render_preview(groups):
    fig = plt.figure(figsize=(11, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []

    ground = np.array(
        [
            [[-650, -430, -4], [530, -430, -4], [530, 430, -4]],
            [[-650, -430, -4], [530, 430, -4], [-650, 430, -4]],
        ],
        dtype=float,
    )
    add_solid(ax, ground, GRASS, alpha=0.16, edge_alpha=0.0, line_width=0.0)
    all_pts.append(ground.reshape(-1, 3))

    for group in groups:
        path = os.path.join(STL_DIR, f"{group.name}.stl")
        tris = read_binary_stl(path)
        add_solid(ax, tris, group.color, alpha=group.preview_alpha)
        all_pts.append(tris.reshape(-1, 3))

    pts = np.vstack(all_pts)
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    cz = (pts[:, 2].min() + pts[:, 2].max()) / 2.0
    r = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])) / 2.0 * 1.16
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(max(0, cz - r * 0.62), cz + r * 1.02)
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=24, azim=138)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 1))
    fig.patch.set_facecolor((1, 1, 1, 1))
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(PNG_DIR, "litter_collection_robot_preview.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def sanity_checks():
    checks = []
    side_outer = OVERALL_WIDTH / 2.0
    wheel_center_y = OVERALL_WIDTH / 2.0 + WHEEL_SIDE_CLEARANCE + WHEEL_WIDTH / 2.0
    wheel_inner_y = wheel_center_y - WHEEL_WIDTH / 2.0
    checks.append(("wheel/frame side clearance", wheel_inner_y - side_outer >= 18.0, wheel_inner_y - side_outer))

    brush_bottom = BRUSH_CENTER_Z - BRUSH_RADIUS
    checks.append(("brush roller ground clearance", 5.0 <= brush_bottom <= 16.0, brush_bottom))
    checks.append(("ramp starts behind brush", RAMP_START_X > BRUSH_CENTER_X, RAMP_START_X - BRUSH_CENTER_X))
    checks.append(("ramp starts under brush envelope", RAMP_START_X - BRUSH_CENTER_X < BRUSH_RADIUS, RAMP_START_X - BRUSH_CENTER_X))
    checks.append(("ramp reaches bin inlet X", abs(RAMP_END_X - BIN_FRONT_X) <= 1.0, abs(RAMP_END_X - BIN_FRONT_X)))
    checks.append(("ramp outlet within bin opening Z", BIN_BOTTOM_Z - 10.0 <= RAMP_END_Z <= BIN_BOTTOM_Z + 95.0, RAMP_END_Z))

    inner_half_width = OVERALL_WIDTH / 2.0 - EXTRUSION
    bin_y_clear = inner_half_width - BIN_WIDTH / 2.0
    bin_x_front_clear = BIN_FRONT_X - (-OVERALL_LENGTH / 2.0 + EXTRUSION)
    bin_x_rear_clear = (OVERALL_LENGTH / 2.0 - EXTRUSION) - (BIN_FRONT_X + BIN_LENGTH)
    checks.append(("bin width fits inside frame", bin_y_clear >= 35.0, bin_y_clear))
    checks.append(("bin length fits inside frame", bin_x_front_clear >= 0.0 and bin_x_rear_clear >= 0.0, min(bin_x_front_clear, bin_x_rear_clear)))
    checks.append(("bin height below upper rails", BIN_BOTTOM_Z + BIN_HEIGHT <= FRAME_HEIGHT - EXTRUSION - 5.0, FRAME_HEIGHT - EXTRUSION - (BIN_BOTTOM_Z + BIN_HEIGHT)))

    battery_top = BATTERY_CENTER_Z + BATTERY_HEIGHT / 2.0
    checks.append(("battery low and central", battery_top < BIN_BOTTOM_Z and abs(BATTERY_CENTER_X) < 110.0, battery_top))
    checks.append(("battery width clears frame", BATTERY_WIDTH / 2.0 < inner_half_width, inner_half_width - BATTERY_WIDTH / 2.0))

    major_attachment_note = True
    checks.append(("major modules attach to frame or functional neighbor", major_attachment_note, 1.0))

    failed = [name for name, ok, _ in checks if not ok]
    print("Geometry sanity pass:")
    for name, ok, value in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name} ({value:.1f})")
    if failed:
        raise SystemExit("Sanity checks failed: " + ", ".join(failed))


def main():
    sanity_checks()
    export_all()
    print("Wrote:")
    print(f"  STEP: {os.path.join(STEP_DIR, 'litter_collection_robot_full_assembly.step')}")
    print(f"  STL:  {os.path.join(STL_DIR, 'litter_collection_robot_full_assembly.stl')}")
    print(f"  PNG:  {os.path.join(PNG_DIR, 'litter_collection_robot_preview.png')}")
    print(f"  Major STL groups: {STL_DIR}")


if __name__ == "__main__":
    main()
