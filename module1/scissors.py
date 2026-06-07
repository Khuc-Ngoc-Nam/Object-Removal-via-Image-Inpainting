import cv2
import numpy as np

# Cấu hình đường dẫn ảnh thực nghiệm (Ví dụ từ tập dữ liệu COCO)
IMAGE_PATH = 'module1/p1.jpg' 

# 1. Khởi tạo và cấu hình bộ trọng số Intelligent Scissors
scissors = cv2.segmentation_IntelligentScissorsMB()
scissors.setWeights(0.4, 0.3, 0.3)

# 2. Đọc ảnh và khởi tạo các ma trận lưu trữ thông tin
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Không thể đọc được ảnh tại đường dẫn: {IMAGE_PATH}")

h, w = img.shape[:2]
img_display = img.copy()

# Cung cấp ảnh gốc cho thuật toán Intelligent Scissors
scissors.applyImage(img)

# Khởi tạo ma trận mặt nạ nhị phân (Binary Mask) đầu ra cho Module Inpainting
binary_mask = np.zeros((h, w), dtype=np.uint8)

# Danh sách lưu trữ tất cả các điểm mốc (anchors) mà người dùng đã click
all_points = []
is_map_built = False
is_drawing = False

def get_smart_contour(start_pt, end_pt, w, h):
    # Kiểm tra xem 2 điểm có cùng nằm trên 1 cạnh biên của ảnh không
    on_same_boundary = False
    if (start_pt[0] == 0 and end_pt[0] == 0) or \
       (start_pt[0] == w - 1 and end_pt[0] == w - 1) or \
       (start_pt[1] == 0 and end_pt[1] == 0) or \
       (start_pt[1] == h - 1 and end_pt[1] == h - 1):
        on_same_boundary = True

    if on_same_boundary:
        # Nếu cùng nằm trên biên, tạo một đường thẳng ôm sát mép ảnh
        dist = max(abs(end_pt[0] - start_pt[0]), abs(end_pt[1] - start_pt[1]))
        if dist > 0:
            xs = np.linspace(start_pt[0], end_pt[0], dist + 1, dtype=np.int32)
            ys = np.linspace(start_pt[1], end_pt[1], dist + 1, dtype=np.int32)
            return np.column_stack((xs, ys)).reshape(-1, 1, 2)
        else:
            return np.array([[[end_pt[0], end_pt[1]]]], dtype=np.int32)
    else:
        # Nếu không, dùng thuật toán hít biên thông thường
        return scissors.getContour(end_pt)

def mouse_callback(event, x, y, flags, param):
    global is_map_built, img_display, all_points, binary_mask, is_drawing
    
    # Tính năng Hít Biên Ảnh (Boundary Snapping)
    # Nếu chuột tiến sát lề ảnh (cách lề <= 15 pixels), ép toạ độ dính chặt vào lề
    SNAP_MARGIN = 15
    if x < SNAP_MARGIN: x = 0
    elif x > w - 1 - SNAP_MARGIN: x = w - 1
    
    if y < SNAP_MARGIN: y = 0
    elif y > h - 1 - SNAP_MARGIN: y = h - 1
    
    current_point = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_drawing:
            # SỰ KIỆN 1: Click lần đầu để bắt đầu đi dây (Magnetic Lasso)
            is_drawing = True
            all_points.append(current_point)
            
            scissors.buildMap(current_point)
            is_map_built = True
            
            img_display = img.copy()
            cv2.circle(img_display, current_point, 4, (0, 0, 255), -1)
            cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
        else:
            # Đang vẽ: Click lại để kết thúc (nếu gần điểm đầu) hoặc buộc chốt điểm (manual anchor)
            # Kiểm tra khoảng cách tới điểm xuất phát
            dist_to_start = np.hypot(current_point[0] - all_points[0][0], current_point[1] - all_points[0][1])
            
            if dist_to_start < 15 and len(all_points) > 2:
                # Đóng loop
                scissors.buildMap(all_points[-1])
                closing_contour = get_smart_contour(all_points[-1], all_points[0], w, h)
                
                cv2.polylines(img, [closing_contour], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.polylines(binary_mask, [closing_contour], isClosed=False, color=255, thickness=2)
                
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(binary_mask, contours, color=255)
                
                print("--> Đã khép kín vật thể thành công! Xuất mặt nạ nhị phân...")
                is_drawing = False
                is_map_built = False
                all_points.clear()
                
                cv2.imshow("Generated Binary Mask (Input for Module 2)", binary_mask)
                img_display = img.copy()
                cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
            else:
                # Buộc chốt điểm mốc thủ công tại vị trí click (manual anchor)
                all_points.append(current_point)
                
                # Lấy contour từ điểm mốc trước đó đến điểm hiện tại (hỗ trợ bám biên)
                fixed_contour = get_smart_contour(all_points[-2], current_point, w, h)
                
                # Vẽ cố định đường biên này
                cv2.polylines(img, [fixed_contour], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.polylines(binary_mask, [fixed_contour], isClosed=False, color=255, thickness=2)
                
                # Build map tại điểm hiện tại để đi dây tiếp
                scissors.buildMap(current_point)
                
                img_display = img.copy()
                for pt in all_points:
                    cv2.circle(img_display, pt, 4, (0, 0, 255), -1)
                cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
                
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing and is_map_built:
        
        # Nếu chuột vừa chạm lề (bị snap) mà mốc trước đó chưa nằm trên lề,
        # tự động chốt một điểm tại mép ảnh để bắt đầu bám lề (ôm khít khung hình).
        x_snapped = (current_point[0] == 0 or current_point[0] == w - 1)
        y_snapped = (current_point[1] == 0 or current_point[1] == h - 1)
        
        if x_snapped or y_snapped:
            last_pt = all_points[-1]
            on_same = False
            if (current_point[0] == 0 and last_pt[0] == 0) or \
               (current_point[0] == w - 1 and last_pt[0] == w - 1) or \
               (current_point[1] == 0 and last_pt[1] == 0) or \
               (current_point[1] == h - 1 and last_pt[1] == h - 1):
                on_same = True
                
            if not on_same:
                # Ép chốt điểm mốc tại lề
                all_points.append(current_point)
                fixed_contour = scissors.getContour(current_point)
                
                cv2.polylines(img, [fixed_contour], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.polylines(binary_mask, [fixed_contour], isClosed=False, color=255, thickness=2)
                
                scissors.buildMap(current_point)
                
                img_display = img.copy()
                for pt in all_points:
                    cv2.circle(img_display, pt, 4, (0, 0, 255), -1)
                cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
                return

        # Lấy contour động từ điểm mốc cuối cùng đến vị trí chuột (hỗ trợ bám biên)
        dynamic_contour = get_smart_contour(all_points[-1], current_point, w, h)
        
        # Tính năng Path Cooling: Tự động chốt điểm mốc khi đoạn dây đủ dài
        COOLING_THRESHOLD = 60 
        FREEZE_MARGIN = 30     
        
        if dynamic_contour is not None and len(dynamic_contour) > COOLING_THRESHOLD:
            freeze_index = len(dynamic_contour) - FREEZE_MARGIN
            
            new_seed_arr = dynamic_contour[freeze_index].reshape(-1)
            new_seed = (int(new_seed_arr[0]), int(new_seed_arr[1]))
            
            all_points.append(new_seed)
            
            fixed_contour = dynamic_contour[:freeze_index+1]
            
            cv2.polylines(img, [fixed_contour], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.polylines(binary_mask, [fixed_contour], isClosed=False, color=255, thickness=2)
            
            scissors.buildMap(new_seed)
            dynamic_contour = get_smart_contour(new_seed, current_point, w, h)
            
        # Cập nhật hiển thị (Preview đường động và các seed points)
        img_display = img.copy()
        if dynamic_contour is not None:
            cv2.polylines(img_display, [dynamic_contour], isClosed=False, color=(0, 255, 0), thickness=2)
            
        # Nếu chuột tiến lại gần điểm xuất phát, vẽ vòng tròn báo hiệu có thể kết thúc
        dist_to_start = np.hypot(current_point[0] - all_points[0][0], current_point[1] - all_points[0][1])
        if dist_to_start < 15 and len(all_points) > 2:
            cv2.circle(img_display, all_points[0], 8, (255, 0, 0), 2)
            
        for pt in all_points:
            cv2.circle(img_display, pt, 4, (0, 0, 255), -1)
            
        cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)

# 3. Khởi tạo cửa sổ tương tác
cv2.namedWindow("Intelligent Scissors Pipeline - Group 25")
cv2.setMouseCallback("Intelligent Scissors Pipeline - Group 25", mouse_callback)

print("=== HƯỚNG DẪN TƯƠNG TÁC (MAGNETIC LASSO) ===")
print("1. Click CHUỘT TRÁI một lần để bắt đầu đi dây.")
print("2. KHÔNG CẦN GIỮ CHUỘT, chỉ cần di chuyển chuột dọc theo biên vật thể, thuật toán sẽ tự động hít và chốt điểm (Path Cooling).")
print("3. Nếu đường hít bị lệch, Click CHUỘT TRÁI để chủ động chốt điểm mốc tại vị trí hiện tại.")
print("4. Để kết thúc, di chuột về GẦN ĐIỂM XUẤT PHÁT (sẽ hiện vòng tròn xanh) và Click CHUỘT TRÁI để khép kín vùng.")
print("5. Nhấn phím bất kỳ để thoát.")

cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()