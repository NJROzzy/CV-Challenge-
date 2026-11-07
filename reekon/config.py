"""Shared execution defaults; algorithm-specific calibration stays beside its formula.

Values match the original implementation. Widths are processing pixels, rates are Hz,
and registration tolerances are fractions of the full-resolution frame diagonal.
"""
SAMPLE_FPS = 10.0
DETECTION_WIDTH = 480
REGISTRATION_WIDTH = 640
KEYFRAME_WIDTH = 776
ROUND_TRIP_TOLERANCE = 0.02
MIN_PATCH_NCC = 0.5
JPEG_QUALITY = 90
HAND_THRESHOLD = 0.15
VIDEO_EXTS = frozenset({'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'})
DETECTORS = ('classical', 'hand-assisted')
