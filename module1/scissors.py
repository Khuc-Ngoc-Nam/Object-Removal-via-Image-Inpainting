import cv2
import numpy as np

# Cấu hình đường dẫn ảnh thực nghiệm (Ví dụ từ tập dữ liệu COCO)
IMAGE_PATH = 'p1.jpg' 

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

def mouse_callback(event, x, y, flags, param):
    global is_map_built, img_display, all_points, binary_mask, is_drawing
    
    # SỰ KIỆN 1: Click và giữ chuột trái để bắt đầu vẽ
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        current_point = (x, y)
        all_points.append(current_point)
        
        # Xây dựng lại bản đồ chi phí với điểm gốc mới nhất
        scissors.buildMap(current_point)
        is_map_built = True
        
        img_display = img.copy()
        cv2.circle(img_display, current_point, 4, (0, 0, 255), -1)
        cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
        
    # SỰ KIỆN 2: Di chuyển chuột (Live-Wire) để hít biên theo thời gian thực
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing and is_map_built:
        current_point = (x, y)
        
        # Lấy contour động (optimal path) từ điểm mốc cuối cùng đến vị trí chuột
        dynamic_contour = scissors.getContour(current_point)
        
        # Tính năng Path Cooling: 
        # Khi đường dây đủ dài, phần đuôi gần điểm gốc đã "nguội" và ổn định.
        # Ta sẽ đóng băng phần này và tự động sinh điểm mốc (seed point) mới
        # ngay TRÊN ĐƯỜNG BIÊN HÍT (optimal path) chứ không phải ở vị trí chuột.
        COOLING_THRESHOLD = 60 # Số lượng điểm (pixel) tối thiểu trên đường động
        FREEZE_MARGIN = 30     # Điểm mốc mới sẽ cách chuột một khoảng an toàn
        
        if dynamic_contour is not None and len(dynamic_contour) > COOLING_THRESHOLD:
            freeze_index = len(dynamic_contour) - FREEZE_MARGIN
            
            # Lấy toạ độ của điểm trên đường viền động để làm seed point mới
            new_seed_arr = dynamic_contour[freeze_index].reshape(-1)
            new_seed = (int(new_seed_arr[0]), int(new_seed_arr[1]))
            
            all_points.append(new_seed)
            
            # Trích xuất đoạn viền đã đóng băng
            fixed_contour = dynamic_contour[:freeze_index+1]
            
            # Vẽ vĩnh viễn đoạn này lên ảnh và mask
            cv2.polylines(img, [fixed_contour], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.polylines(binary_mask, [fixed_contour], isClosed=False, color=255, thickness=2)
            
            # Xây dựng lại bản đồ chi phí từ điểm mốc mới
            scissors.buildMap(new_seed)
            
            # Cập nhật lại đường viền động từ mốc mới tới vị trí chuột hiện tại
            dynamic_contour = scissors.getContour(current_point)
            
        # Cập nhật hiển thị (Preview đường động và các seed points)
        img_display = img.copy()
        if dynamic_contour is not None:
            cv2.polylines(img_display, [dynamic_contour], isClosed=False, color=(0, 255, 0), thickness=2)
            
        for pt in all_points:
            cv2.circle(img_display, pt, 4, (0, 0, 255), -1)
            
        cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
            
    # SỰ KIỆN 3: Nhả chuột trái để KHÉP KÍN VÙNG và TRÍCH XUẤT MASK
    elif event == cv2.EVENT_LBUTTONUP and is_drawing:
        is_drawing = False
        if len(all_points) > 2:
            # Nối điểm cuối cùng quay trở lại điểm xuất phát
            scissors.buildMap(all_points[-1]) 
            closing_contour = scissors.getContour(all_points[0])
            
            # Vẽ đoạn khép kín cuối cùng
            cv2.polylines(img, [closing_contour], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.polylines(binary_mask, [closing_contour], isClosed=False, color=255, thickness=2)
            
            # Tạo đặc mặt nạ
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillPoly(binary_mask, contours, color=255)
            
            print("--> Đã khép kín vật thể thành công! Xuất mặt nạ nhị phân...")
            is_map_built = False
            all_points.clear()
            
            # Hiển thị kết quả mặt nạ cuối cùng
            cv2.imshow("Generated Binary Mask (Input for Module 2)", binary_mask)
            img_display = img.copy()
            cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)

# 3. Khởi tạo cửa sổ tương tác
cv2.namedWindow("Intelligent Scissors Pipeline - Group 25")
cv2.setMouseCallback("Intelligent Scissors Pipeline - Group 25", mouse_callback)

print("=== HƯỚNG DẪN TƯƠNG TÁC ===")
print("1. Click và GIỮ CHUỘT TRÁI rồi di chuyển để khoanh vùng vật thể.")
print("2. Thuật toán sẽ tự động hít biên (Live-wire) theo đường di chuột.")
print("3. NHẢ CHUỘT TRÁI để tự động khép kín vùng và tạo Binary Mask.")
print("4. Nhấn phím bất kỳ để thoát.")

cv2.imshow("Intelligent Scissors Pipeline - Group 25", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()