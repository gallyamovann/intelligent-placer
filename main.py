from detection import get_fill_masks
import cv2
from matplotlib import pyplot as plt

PNG_FORMAT = "*.png"
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
THICKNESS = 5
CONTOUR_IDX = -1
BORDER = 10


def demonstration_items(path_items):
    paths = []
    for p_item in path_items.glob(PNG_FORMAT):
        paths.append(p_item)

    for p in paths:
        print(p)
        # загрузка изображения
        image = cv2.imread(str(p))
        img, binary, polys, _ = get_fill_masks(image.copy())

        x, y, w, h = cv2.boundingRect(polys[0])
        # ROI (Region of interest)
        roi = img[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
        roi_binary = binary[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
        contours = img.copy()
        cv2.drawContours(contours, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)

        imgs = []

        imgs.append(image)
        imgs.append(binary)
        imgs.append(roi)
        imgs.append(roi_binary)
        imgs.append(contours[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER])
        imgs.append(contours)

        _, axs = plt.subplots(1, 6, figsize=(12, 12))
        axs = axs.flatten()
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
    paths = []
    for p_test in path_tests.glob(PNG_FORMAT):
        paths.append(p_test)
    for p in paths:
        # загрузка изображения
        image = cv2.imread(str(p))
        img, binary, polys, _ = get_fill_masks(image.copy())
        bbox = img.copy()

        contours = img.copy()
        cv2.drawContours(contours, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)
        imgs_items = []
        for poly in polys:
            x, y, w, h = cv2.boundingRect(poly)
            bbox = cv2.rectangle(bbox, (x, y), (x + w, y + h), RGB_BLACK, THICKNESS)
            cv2.drawContours(img, polys, CONTOUR_IDX, RGB_WHITE, THICKNESS)
            # ROI (Region of interest)
            roi = img[y - BORDER: y + h + BORDER, x - BORDER: x + w + BORDER]
            imgs_items.append(roi)

        imgs = []

        imgs.append(image)
        imgs.append(bbox)
        imgs.append(binary)
        imgs.append(contours)

        _, axs = plt.subplots(1, 4, figsize=(12, 12))
        axs = axs.flatten()
        axs[0].set_title("Image")
        axs[1].set_title("With bbox")
        axs[2].set_title("Binary image")
        axs[3].set_title("With contours")
        for im, ax in zip(imgs, axs):
            ax.imshow(im)
        plt.show()
        _, axs_p = plt.subplots(1, len(polys), figsize=(12, 12))
        if len(polys) > 1:
            axs_p = axs_p.flatten()
            for im, ax in zip(imgs_items, axs_p):
                ax.imshow(im)
            for i, ax in enumerate(axs_p.ravel()):
                ax.set_title("contour #{}".format(i))
        else:
            plt.title("contour")
            plt.imshow(imgs_items[0])
        plt.show()


def run_tests(path_tests):
    paths = []
    for p_test in path_tests.glob(PNG_FORMAT):
        paths.append(p_test)
    result_area = []
    result_radius = []
    for p in paths:
        radius = []
        image = cv2.imread(str(p))
        _, _, _, cnts = get_fill_masks(image.copy())
        areas = [cv2.contourArea(cnt) for cnt in cnts]
        sum_obj_area = 0
        for cnt in cnts:
            (_, _), r = cv2.minEnclosingCircle(cnt)
            radius.append(2.0 * r)
        is_fit = True
        for i in range(len(areas) - 1):
            sum_obj_area += areas[i]
            if radius[i] > radius[-1]:
                is_fit = False
        if sum_obj_area < areas[-1]:
            result_area.append("True")
        else:
            result_area.append("False")
        result_radius.append(is_fit)
    return result_area, result_radius