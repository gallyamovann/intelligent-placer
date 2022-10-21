import cv2

K_SIZE = 7
MIN_AREA = 1000
THRESHOLD_LOW = 100
THRESHOLD_HIGH = 400
APPROX_CURVE = 0.000001
RGB = (255, 255, 255)
THICKNESS = 5
CONTOUR_IDX = -1

def get_fill_masks(image):
    #преобразуем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ищем границы
    edged = cv2.Canny(gray, THRESHOLD_LOW, THRESHOLD_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K_SIZE, K_SIZE))
    #морфологическая операция закрытия
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #находим контуры
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks_coords = []
    for cnt in contours: #цикл по всем контурам
        area = cv2.contourArea(cnt)
        # обнаруженные небольшие дефекты исключаем, задав минимальную площадь
        if area > MIN_AREA:
            peri = cv2.arcLength(cnt, True)
            # аппроксимируем найденный контур многоугольником
            approx = cv2.approxPolyDP(cnt, APPROX_CURVE * peri, True)
            masks_coords.append(approx)
            cv2.drawContours(closed, [approx], CONTOUR_IDX, RGB, THICKNESS)
            cv2.fillPoly(closed, pts=[approx], color=RGB)
    return closed, masks_coords

    

