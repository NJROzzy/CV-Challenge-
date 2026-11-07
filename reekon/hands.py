"""AI-assisted detection strategy: classical tape detection + GPU hand-pose evidence.

Uses NVIDIA's pretrained Pose-ResNet18-Hand (21 hand keypoints) through jetson-inference /
TensorRT. It was not trained on tape measures and is not fine-tuned here. Its role:

  * a hand at the visible hook end is evidence that the hook is being placed against an edge,
    so that observation's confidence is raised (feeds the hook vote in reekon.events);
  * record hand keypoint counts per frame for inspection.

Rejected rule (measured on Cabinet-1.MOV): "hand at the in-frame end => that end is the housing".
It fired on 5/66 frames, all of them hook-placement frames where the hand presses the hook to the
cabinet edge, so it would have turned correct hook sightings into wrong ones.

Requires the system Python that has `jetson_inference` (python3 on JetPack), not .venv.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .config import DETECTION_WIDTH, HAND_THRESHOLD
from .detect import ClassicalTapeDetector
from .video import resize_to_width


class HandAssistedDetector:
    name = 'hand-assisted'

    def __init__(self, models_dir=Path.home() / 'jetson-inference' / 'data', threshold=HAND_THRESHOLD, proc_width=DETECTION_WIDTH):
        from jetson_inference import poseNet
        from jetson_utils import cudaFromNumpy
        self.classical = ClassicalTapeDetector(proc_width=proc_width)
        self.proc_width = proc_width
        cwd = os.getcwd()
        os.chdir(models_dir)  # jetson-inference resolves 'networks/...' relative to the CWD
        try:
            self.net = poseNet('resnet18-hand', threshold=threshold)
        finally:
            os.chdir(cwd)
        self._to_cuda = cudaFromNumpy

    def hands(self, image):
        small, scale = resize_to_width(image, self.proc_width)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        poses = self.net.Process(self._to_cuda(rgb), overlay='none')
        return [np.array([[k.x / scale, k.y / scale] for k in p.Keypoints]) for p in poses if len(p.Keypoints)]

    def detect(self, image, frame_index=-1, t=0.0):
        obs = self.classical.detect(image, frame_index, t)
        if obs is None:
            return None
        hands = self.hands(image)
        obs.detector = self.name
        obs.evidence['hand_keypoint_sets'] = len(hands)
        if obs.hook.visible and hands:
            reach = 6 * obs.blade_width_px
            near = min(np.min(np.linalg.norm(h - obs.hook.xy, axis=1)) for h in hands)
            if near <= reach:
                obs.evidence['hand_at_hook'] = True
                obs.confidence = round(min(1.0, obs.confidence * 1.1), 3)
        return obs
