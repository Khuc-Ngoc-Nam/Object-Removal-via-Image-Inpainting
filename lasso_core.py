from __future__ import annotations

import cv2
import numpy as np


class LassoCore:
    def __init__(self, target_max_side: int = 768, snap_margin: int = 15):
        self.scissors = cv2.segmentation_IntelligentScissorsMB()
        self.scissors.setEdgeFeatureCannyParameters(threshold1=50, threshold2=100)
        self.scissors.setWeights(0.3, 0.3, 0.4)
        self.target_max_side = target_max_side
        self.snap_margin = snap_margin
        self.orig_image = None
        self.orig_h = 0
        self.orig_w = 0
        self.proc_h = 0
        self.proc_w = 0
        self.scale = 1.0

    def apply_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        self.orig_image = img
        self.orig_h, self.orig_w = img.shape[:2]
        self.scale = min(1.0, self.target_max_side / max(self.orig_h, self.orig_w))
        self.proc_w = max(1, int(round(self.orig_w * self.scale)))
        self.proc_h = max(1, int(round(self.orig_h * self.scale)))

        img_resized = cv2.resize(img, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
        self.scissors.applyImage(img_resized)
        return self.orig_w, self.orig_h

    def _clamp_original(self, point):
        x = max(0, min(int(round(point[0])), self.orig_w - 1))
        y = max(0, min(int(round(point[1])), self.orig_h - 1))
        return x, y

    def _snap_original(self, point):
        x, y = self._clamp_original(point)
        if x <= self.snap_margin:
            x = 0
        elif x >= self.orig_w - 1 - self.snap_margin:
            x = self.orig_w - 1

        if y <= self.snap_margin:
            y = 0
        elif y >= self.orig_h - 1 - self.snap_margin:
            y = self.orig_h - 1

        return x, y

    def _to_proc(self, point):
        x, y = self._snap_original(point)
        proc_x = max(0, min(int(round(x * self.scale)), self.proc_w - 1))
        proc_y = max(0, min(int(round(y * self.scale)), self.proc_h - 1))
        return proc_x, proc_y

    def _to_original_points(self, contour):
        points = []
        for pt in contour.reshape(-1, 2):
            x = int(round(pt[0] / self.scale)) if self.scale else int(pt[0])
            y = int(round(pt[1] / self.scale)) if self.scale else int(pt[1])
            points.append({"x": self._clamp_original((x, y))[0], "y": self._clamp_original((x, y))[1]})
        return points

    def build_map(self, point):
        self.scissors.buildMap(self._to_proc(point))

    def _line_contour(self, start, end):
        s_x, s_y = start
        e_x, e_y = end
        dist = max(abs(e_x - s_x), abs(e_y - s_y))
        if dist <= 0:
            return np.array([[[e_x, e_y]]], dtype=np.int32)

        xs = np.linspace(s_x, e_x, dist + 1, dtype=np.int32)
        ys = np.linspace(s_y, e_y, dist + 1, dtype=np.int32)
        return np.column_stack((xs, ys)).reshape(-1, 1, 2)

    def get_smart_contour(self, start_pt, end_pt):
        s_x, s_y = self._to_proc(start_pt)
        e_x, e_y = self._to_proc(end_pt)

        on_same_boundary = (
            (s_x == 0 and e_x == 0)
            or (s_x == self.proc_w - 1 and e_x == self.proc_w - 1)
            or (s_y == 0 and e_y == 0)
            or (s_y == self.proc_h - 1 and e_y == self.proc_h - 1)
        )

        if on_same_boundary:
            contour = self._line_contour((s_x, s_y), (e_x, e_y))
        else:
            contour = self.scissors.getContour((e_x, e_y))
            if contour is None or len(contour) == 0:
                contour = self._line_contour((s_x, s_y), (e_x, e_y))

        return self._to_original_points(contour)

    def generate_mask(self, points):
        if not points:
            return np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)

        snapped = [self._snap_original((pt["x"], pt["y"])) for pt in points]
        contour = np.array(snapped, dtype=np.int32).reshape(-1, 1, 2)
        binary_mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [contour], 255)
        return binary_mask
