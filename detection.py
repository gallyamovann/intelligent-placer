import cv2

K_SIZE = (7, 7)
BLUR = (1, 1)
SIGMA_BLUR = 0
MIN_AREA = 1000
THRESHOLD_LOW = 100
THRESHOLD_HIGH = 400
APPROX_CURVE = 0.000001
RGB = (0, 0, 0)
THICKNESS = 5
CONTOUR_IDX = -1

def get_fill_masks(image):
    height, width = image.shape[:2]
    # ищем границы
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR, SIGMA_BLUR)
    edged = cv2.Canny(blur, THRESHOLD_LOW, THRESHOLD_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, K_SIZE)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > (height - 200) * (width - 200):
            x, y, w, h = cv2.boundingRect(cnt)
            image = image[y + 50:y + h - 50, x + 50:x + w - 50]
            closed = closed[y + 50:y + h - 50, x + 50:x + w - 50]
            contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, binary_img = cv2.threshold(image, 150, 250, cv2.THRESH_BINARY)
    binary = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    # зададим список для найденных масок
    masks_coords = []
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # обнаруженные небольшие дефекты исключаем, задав минимальную площадь
        if area > MIN_AREA:
            cnts.append(cnt)
            peri = cv2.arcLength(cnt, True)
            # аппроксимируем найденный контур многоугольником
            approx = cv2.approxPolyDP(cnt, APPROX_CURVE * peri, True)
            masks_coords.append(approx)
            cv2.drawContours(binary, [approx], CONTOUR_IDX, RGB, THICKNESS)
            cv2.fillPoly(binary, pts=[approx], color=RGB)
    return image, binary, masks_coords, cnts