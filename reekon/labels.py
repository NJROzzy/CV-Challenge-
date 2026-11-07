"""User-facing labels for validated measurements: type (height / width / diagonal), the direction the
tape ran, and - where it can be established - the position relative to other measurements of the
same type (left / right for heights, top / bottom for widths).

Type and direction come from the measurement's own endpoints in its reading frame, assuming the
PoV camera is roughly upright (head-mounted footage; roll is not estimated):
  * within 30 deg of vertical  -> height;  within 30 deg of horizontal -> width;  otherwise diagonal.

Relative position is only assigned when two measurements of the same type can be compared in ONE
view: one measurement's midpoint is carried into the other's reading frame with the validated
mapping (round trip + appearance checks). Without a validated mapping no position is invented.
"""
import math

import numpy as np


def kind_and_direction(m):
    hook = np.array([m.hook.x, m.hook.y])
    housing = np.array([m.housing.x, m.housing.y])
    dx, dy = housing - hook
    from_horizontal = math.degrees(math.atan2(abs(dy), abs(dx)))
    if from_horizontal >= 60:
        return 'height', ('top → bottom' if dy > 0 else 'bottom → top')
    if from_horizontal <= 30:
        return 'width', ('left → right' if dx > 0 else 'right → left')
    vertical = 'top' if dy > 0 else 'bottom'
    horizontal = 'left' if dx > 0 else 'right'
    end_v = 'bottom' if vertical == 'top' else 'top'
    end_h = 'right' if horizontal == 'left' else 'left'
    return 'diagonal', f'{vertical}-{horizontal} → {end_v}-{end_h}'


def _midpoint(m):
    return [(m.hook.x + m.housing.x) / 2, (m.hook.y + m.housing.y) / 2]


def relative_positions(measurements, registrar, max_pairs=40):
    """{measurement id: 'left'/'right'/'top'/'bottom'/'middle'} for measurements whose order against
    another measurement of the same type was established through a validated mapping."""
    by_kind = {}
    for m in measurements:
        by_kind.setdefault(kind_and_direction(m)[0], []).append(m)
    positions, tried = {}, 0
    for kind, group in by_kind.items():
        if kind == 'diagonal' or len(group) < 2:
            continue
        axis = 0 if kind == 'height' else 1          # heights ordered left-right, widths top-bottom
        names = ('left', 'right') if kind == 'height' else ('top', 'bottom')
        # score[m] = how many same-type measurements were found before it along the axis
        before = {m.id: set() for m in group}
        compared = set()
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if tried >= max_pairs:
                    break
                tried += 1
                pts, ok, _, _ = registrar.map_validated(b.canvas_frame, a.canvas_frame, [_midpoint(b)])
                if pts is None or not ok.all():
                    continue
                pa, pb = _midpoint(a)[axis], pts[0][axis]
                if abs(pa - pb) < 1e-6:
                    continue
                (first, second) = (a, b) if pa < pb else (b, a)
                before[second.id].add(first.id)
                compared |= {a.id, b.id}
        members = [m for m in group if m.id in compared]
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda m: len(before[m.id]))
        for rank, m in enumerate(ranked):
            if rank == 0:
                positions[m.id] = names[0]
            elif rank == len(ranked) - 1:
                positions[m.id] = names[1]
            else:
                positions[m.id] = 'middle'
    return positions


def label_measurements(measurements, registrar):
    """Attach m.label = {'kind', 'direction', 'position', 'text'} to every validated measurement."""
    valid = [m for m in measurements if m.status == 'validated']
    positions = relative_positions(valid, registrar) if registrar is not None else {}
    for m in valid:
        kind, direction = kind_and_direction(m)
        pos = positions.get(m.id)
        name = kind.capitalize() + (f' ({pos})' if pos else '')
        m.label = {'kind': kind, 'direction': direction, 'position': pos,
                   'text': f'#{m.id} {name}', 'detail': f'{name}, measured {direction}'}
    return valid
