import cv2
import numpy as np
import shapely.ops as so
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate, rotate

MIN_AREA = 1000  # площадь, меньше которой обнаруженные маски будут считаться дефектными


def get_edges(image):
    """
    Обнаруживаем границы изображения с помощью детектора Кэнни и морфологических операций
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(blur, 100, 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


def get_contours(image):
    """
    Строим контуры на изображении
    """
    closed = get_edges(image)
    contours, _ = cv2.findContours(closed,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def place_object(poly, obj, placed_objects,
                 field_size,
                 shift_step=10,
                 rotate_step=10,
                 min_degree=0,
                 max_degree=180):
    """
    Функция размещения объектов во многоугольнике
    """
    min_x, min_y = -int(obj.centroid.x), -int(obj.centroid.y)
    max_x, max_y = np.array(field_size) - np.array([int(obj.centroid.x), int(obj.centroid.y)])
    intersect_placed = True
    # проходим по х-координатам
    for dx in range(min_x, max_x, shift_step):
        # проходим по у-координатам
        for dy in range(min_y, max_y, shift_step):
            translated = translate(obj, dx, dy)
            # вращаем предмет вокруг себя
            for dr in range(min_degree, max_degree, rotate_step):
                rotated = rotate(translated, -dr, origin='centroid')
                # проверяем помещается ли многоугольник в объект
                if poly.contains(rotated):
                    intersect_placed = False
                for placed in placed_objects:
                    # если объект пересекается с другими размещенным объектом
                    # место во многоугольнике не найдено
                    if placed.intersection(rotated).area > 0:
                        intersect_placed = True
                        break
                # если все хорошо, то добавляем его в массив размещенных объектов
                if not intersect_placed:
                    placed_objects.append(rotated)
                    return True
    return False


def check_diameter(radius, polygon_index):
    """
    Функция проверяет, вмещается ли предмет во многоугольник по габаритам
    """
    for i in range(len(radius)):
        if radius[i] > radius[polygon_index]:
            return False
    return True


def check_area(areas, polygon_index):
    """
    Функция проверяет, вмещаются ли предметы во многоугольник по площади
    """
    sum_obj_area = sum(areas)
    if sum_obj_area < 2 * areas[polygon_index]:
        return True
    return False


def plot_image(contours, image):
    """
    Отображаем изображение с масками на многоугольниках и предметах, а также контурах
    """
    new_image = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            approx = cv2.approxPolyDP(cnt, 0.000001 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(new_image, [approx], 0, (0, 0, 0), 10)
            cv2.fillPoly(new_image, pts=[approx], color=(255, 255, 255))
    ret, thresh = cv2.threshold(new_image, 254, 255, cv2.THRESH_BINARY)
    images = [image, new_image, thresh]
    titles = ['Исходное изображение', 'Изображение с контурами', 'Бинарное изображение']
    for i in range(3):
        plt.subplot(3, 1, i + 1), plt.imshow(images[i], vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def run(image_path: str, shift_step=10, rotate_step=5):
    """
    Общая функция, которая обрабатывает полученное изображение и запускает алгоритм расстановки
    """
    if image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        polys = []
        areas = []
        radius = []
        image = cv2.imread(image_path)
        # сортируем предметы в порядке убывания
        contours = sorted(get_contours(image), key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                (_, _), r = cv2.minEnclosingCircle(cnt)
                radius.append(2.0 * r)
                areas.append(area)
                polys.append(cv2.approxPolyDP(cnt, 0.000001 * cv2.arcLength(cnt, True), True))
        # находим многоугольник по наименьшей у-координате
        polygon_index = np.argmin([np.min(p[:, 0, 1]) for p in polys])
        objects = list(map(lambda x: Polygon(x[:, 0, :]),
                           polys[0:polygon_index] + polys[polygon_index + 1:]))
        polygon = Polygon(polys[polygon_index][:, 0, :])
        # проверяем на вместимость по габаритам
        areas_fit = check_area(areas, polygon_index)
        # и по площади
        diameter_fit = check_diameter(radius, polygon_index)
        plot_image(contours, image)
        result = False
        height, width = image.shape[:2]
        # если элементарные проверки пройдены
        if areas_fit and diameter_fit:
            placed_objects = []
            can_fit = True
            for obj in objects:
                # запускаем алгоритм упаковки
                result = place_object(polygon,
                                      make_valid(obj),
                                      placed_objects,
                                      (height, width),
                                      shift_step=shift_step,
                                      rotate_step=rotate_step,
                                      min_degree=0,
                                      max_degree=180)
                if not result:
                    can_fit = False
                    break
            if can_fit:
                plot_placed_objects(placed_objects, polygon)
        return result
    return "Указан неверный путь к изображению"


def plot_placed_objects(placed_objects, polygon):
    """
    Рисуем размещенные предметы
    """
    objs = [obj for obj in placed_objects]
    new_shape = so.unary_union(objs)
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    if type(new_shape) == Polygon:
        xs, ys = new_shape.exterior.xy
        axs.fill(xs, ys, alpha=1, fc='r', ec='none')
    else:
        for geom in new_shape.geoms:
            if type(geom) == LineString:
                xs, ys = geom.xy
                axs.fill(xs, ys, alpha=1, fc='r', ec='none')
            else:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=1, fc='r', ec='none')
    plt.rcParams["figure.figsize"] = [10.00, 10.00]
    plt.rcParams["figure.autolayout"] = True
    x, y = polygon.exterior.xy
    plt.plot(x, y, c="black")
    plt.show()
