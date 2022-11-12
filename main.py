from detection import get_fill_masks
import cv2
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

PNG_FORMAT = "*."
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
THICKNESS = 5
CONTOUR_IDX = -1
BORDER = 10


# получим все изображения необходимого формата из папки 
def get_paths(path):
    paths = []
    for p_test in path.glob(PNG_FORMAT):
        paths.append(p_test)
    return paths


def demonstration_items(path_items):
    paths = get_paths(path_items)
    for p in paths:
        print(p)
        # загрузка изображения
        image = cv2.imread(str(p))
        img, binary, polys, _ = get_fill_masks(image.copy())

        x, y, w, h = cv2.boundingRect(polys[0])
        # обрежем изображение
        roi = img[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
        # обрежем бинарное изображение
        roi_binary = binary[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
        contours = img.copy()
        # нарисуем контуры на исходном изображении
        cv2.drawContours(contours, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)

        imgs = []
        # примеры представления изображений 
        # (исходное, бинарное, обрезанные(+бинарное/с контурами), с контурами)
        imgs.append(image)
        imgs.append(binary)
        imgs.append(roi)
        imgs.append(roi_binary)
        imgs.append(contours[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER])
        imgs.append(contours)

        # покажем результаты
        _, axs = plt.subplots(1, 6, figsize=(12, 12))
        # axs = axs.flatten()
        # отобразим изображения
        for im, ax in zip(imgs, axs):
            ax.imshow(im)
        axs[0].set_title("Image")
        axs[1].set_title("Binary image")
        axs[2].set_title("ROI image")
        axs[3].set_title("ROI binary image")
        axs[4].set_title("With contours (roi)")
        axs[5].set_title("With contours")
        plt.show()


def demonstration_test(path_tests):
    paths = get_paths(path_tests)
    for p in paths:
        # загрузка изображения
        image = cv2.imread(str(p))
        img, binary, polys, _ = get_fill_masks(image.copy())
        bbox = img.copy()
        contours = img.copy()
        cv2.drawContours(contours, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)
        imgs_items = []
        # изобразим контуры на каждом предмете и обрежем затем предметы по bbox
        for poly in polys:
            x, y, w, h = cv2.boundingRect(poly)
            bbox = cv2.rectangle(bbox, (x, y), (x + w, y + h), RGB_BLACK, THICKNESS)
            cv2.drawContours(img, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)
            # ROI (Region of interest)
            roi = img[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
            imgs_items.append(roi)

        imgs = []

        # примеры представления изображений 
        # (исходное, с bbox, бинарное, с контурами)
        imgs.append(image)
        imgs.append(bbox)
        imgs.append(binary)
        imgs.append(contours)

        _, axs = plt.subplots(1, 4, figsize=(12, 12))
        # axs = axs.flatten()
        axs[0].set_title("Image")
        axs[1].set_title("With bbox")
        axs[2].set_title("Binary image")
        axs[3].set_title("With contours")
        # отобразим изображения
        for im, ax in zip(imgs, axs):
            ax.imshow(im)
        plt.show()
        _, axs_p = plt.subplots(1, len(polys), figsize=(12, 12))
        # отобразим каждый предмет в ограничивающем bbox
        if len(polys) > 1:
            # axs_p = axs_p.flatten()
            for im, ax in zip(imgs_items, axs_p):
                ax.imshow(im)
            for i, ax in enumerate(axs_p):
                ax.set_title("contour #{}".format(i))
        else:
            plt.title("contour")
            plt.imshow(imgs_items[0])
        plt.show()


def make_polys_objects(path_tests):
    paths = get_paths(path_tests)
    for p in paths:
        # загрузка изображения
        image = cv2.imread(str(p))
        img, binary, polys, _ = get_fill_masks(image.copy())
        polygon_index = np.argmin([np.min(p[:, 0, 1]) for p in polys])
        objects = list(map(lambda x: Polygon(x[:, 0, :]), polys[0:polygon_index] + polys[polygon_index + 1:]))
        polygon = Polygon(polys[polygon_index][:, 0, :])
        new_shape = so.cascaded_union([objs for objs in objects])
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        if type(new_shape) is not Polygon:
            #  continue
            for geom in new_shape.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=1, fc='r', ec='none')
        else:
            xs, ys = new_shape.exterior.xy
            axs.fill(xs, ys, alpha=1, fc='r', ec='none')
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        x, y = polygon.exterior.xy
        plt.plot(x, y, c="red")
        plt.show()


# сделаем проверку по площади и диаметру объектов в сравнении с многоугольником
def run_tests(path_tests):
    paths = get_paths(path_tests)
    # храним результаты проверки в двух списках ниже
    result_area = []
    result_radius = []
    for p in paths:
        radius = []
        image = cv2.imread(str(p))
        _, _, _, cnts = get_fill_masks(image.copy())
        # считаем площади по контурам
        areas = [cv2.contourArea(cnt) for cnt in cnts]
        sum_obj_area = 0
        # находим минимально ограничивающий круг и его радиус
        for cnt in cnts:
            (_, _), r = cv2.minEnclosingCircle(cnt)
            radius.append(2.0 * r)
        is_fit = True
        # считаем суммарную площадь и сравниваем с площадью многоугольника
        # сравниваем радиус каждого предмета с радиусом многоугольника
        for i in range(len(areas) - 1):
            sum_obj_area += areas[i]
            if radius[i] > radius[-1]:
                is_fit = False
        if sum_obj_area < areas[-1]:
            result_area.append("True")
        else:
            result_area.append("False")
        result_radius.append(is_fit)
        # выводим результат
    return result_area, result_radius
