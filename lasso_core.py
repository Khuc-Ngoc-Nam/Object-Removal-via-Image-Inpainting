import cv2
import numpy as np

class LassoCore:
    def __init__(self, target_proc_size=(512, 512)):
        self.scissors = cv2.segmentation_IntelligentScissorsMB()
        self.scissors.setEdgeFeatureCannyParameters(threshold1=50, threshold2=100)
        self.scissors.setWeights(0.3, 0.3, 0.4)
        self.target_w, self.target_h = target_proc_size
        self.orig_image = None
        self.orig_h, self.orig_w = 0, 0
        self.scale_x = 1.0
        self.scale_y = 1.0

    def apply_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
            
        self.orig_image = img
        self.orig_h, self.orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        self.scale_x = self.target_w / self.orig_w
        self.scale_y = self.target_h / self.orig_h
        
        self.scissors.applyImage(img_resized)
        return self.orig_w, self.orig_h

    def build_map(self, point):
        proc_x = int(point[0] * self.scale_x)
        proc_y = int(point[1] * self.scale_y)
        proc_x = max(0, min(proc_x, self.target_w - 1))
        proc_y = max(0, min(proc_y, self.target_h - 1))
        
        self.scissors.buildMap((proc_x, proc_y))

    def get_smart_contour(self, start_pt, end_pt):
        s_x, s_y = int(start_pt[0] * self.scale_x), int(start_pt[1] * self.scale_y)
        e_x, e_y = int(end_pt[0] * self.scale_x), int(end_pt[1] * self.scale_y)
        
        s_x = max(0, min(s_x, self.target_w - 1))
        s_y = max(0, min(s_y, self.target_h - 1))
        e_x = max(0, min(e_x, self.target_w - 1))
        e_y = max(0, min(e_y, self.target_h - 1))

        on_same_boundary = False
        if (s_x == 0 and e_x == 0) or \
           (s_x == self.target_w - 1 and e_x == self.target_w - 1) or \
           (s_y == 0 and e_y == 0) or \
           (s_y == self.target_h - 1 and e_y == self.target_h - 1):
            on_same_boundary = True

        if on_same_boundary:
            dist = max(abs(e_x - s_x), abs(e_y - s_y))
            if dist > 0:
                xs = np.linspace(s_x, e_x, dist + 1, dtype=np.int32)
                ys = np.linspace(s_y, e_y, dist + 1, dtype=np.int32)
                contour = np.column_stack((xs, ys)).reshape(-1, 1, 2)
            else:
                contour = np.array([[[e_x, e_y]]], dtype=np.int32)
        else:
            contour = self.scissors.getContour((e_x, e_y))
            
        if contour is None:
            return []
            
        return [{"x": int(pt[0][0] / self.scale_x), "y": int(pt[0][1] / self.scale_y)} for pt in contour]

    def generate_mask(self, points):
        small_mask = np.zeros((self.target_h, self.target_w), dtype=np.uint8)
        if not points:
            return np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
            
        np_points = np.array([[int(pt["x"] * self.scale_x), int(pt["y"] * self.scale_y)] for pt in points], dtype=np.int32)
        
        cv2.drawContours(small_mask, [np_points], -1, 255, thickness=-1)
            
        binary_mask = cv2.resize(small_mask, (self.orig_w, self.orig_h), interpolation=cv2.INTER_NEAREST)
        
        kernel_size = max(3, int(self.orig_w / 500)) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        return binary_mask