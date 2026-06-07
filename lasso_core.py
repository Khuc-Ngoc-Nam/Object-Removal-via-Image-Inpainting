import cv2
import numpy as np

class LassoCore:
    def __init__(self):
        self.scissors = cv2.segmentation_IntelligentScissorsMB()
        self.scissors.setWeights(0.43, 0.43, 0.14)
        self.image = None
        self.h = 0
        self.w = 0

    def apply_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        self.image = img
        self.h, self.w = img.shape[:2]
        self.scissors.applyImage(img)
        return self.w, self.h

    def build_map(self, point):
        # point is [x, y]
        pt = (int(point[0]), int(point[1]))
        self.scissors.buildMap(pt)

    def get_smart_contour(self, start_pt, end_pt):
        start_pt = (int(start_pt[0]), int(start_pt[1]))
        end_pt = (int(end_pt[0]), int(end_pt[1]))
        
        on_same_boundary = False
        if (start_pt[0] == 0 and end_pt[0] == 0) or \
           (start_pt[0] == self.w - 1 and end_pt[0] == self.w - 1) or \
           (start_pt[1] == 0 and end_pt[1] == 0) or \
           (start_pt[1] == self.h - 1 and end_pt[1] == self.h - 1):
            on_same_boundary = True

        if on_same_boundary:
            dist = max(abs(end_pt[0] - start_pt[0]), abs(end_pt[1] - start_pt[1]))
            if dist > 0:
                xs = np.linspace(start_pt[0], end_pt[0], dist + 1, dtype=np.int32)
                ys = np.linspace(start_pt[1], end_pt[1], dist + 1, dtype=np.int32)
                contour = np.column_stack((xs, ys)).reshape(-1, 1, 2)
            else:
                contour = np.array([[[end_pt[0], end_pt[1]]]], dtype=np.int32)
        else:
            contour = self.scissors.getContour(end_pt)
            
        # Convert numpy contour to list of dicts for JSON
        if contour is None:
            return []
            
        pts = []
        for i in range(len(contour)):
            pts.append({"x": int(contour[i][0][0]), "y": int(contour[i][0][1])})
        return pts

    def generate_mask(self, points):
        # points is a list of [x, y] coordinates forming the closed boundary
        binary_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Draw the boundary on the mask
        np_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(binary_mask, [np_points], isClosed=True, color=255, thickness=2)
        
        # Fill the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(binary_mask, contours, color=255)
        
        return binary_mask
