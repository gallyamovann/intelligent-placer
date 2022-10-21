from detection import get_fill_masks
import cv2
from matplotlib import pyplot as plt

img_path = "img04.png"
file_path = "data/test/"

# загрузка изображения
image = cv2.imread(file_path+img_path)
image, polys = get_fill_masks(image.copy())
plt.imshow(image)
plt.show()