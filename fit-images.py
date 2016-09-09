import cv2
import pathlib as plb
import numpy as np

def crop_image(img, cr_shape):
    cr_n, cr_m = cr_shape
    out_image = np.zeros((cr_n, cr_m), dtype=np.uint8)
    n, m = img.shape
    i_s = (n - cr_n) // 2
    j_s = (m - cr_m) // 2
    for i in range(cr_n):
        for j in range(cr_m):
            out_image[i][j] = img[i_s + i][j_s + j]
    return out_image

#in_path = plb.Path('101_ObjectCategories/Faces_easy')
#out_path  = plb.Path('img/resized/Faces_easy_388_width')

in_path = plb.Path('101_ObjectCategories/Motorbikes')
out_path  = plb.Path('./img/resized/Motorbikes_height_388')

imgs = [(filename.stem, cv2.imread(filename.as_posix(), cv2.CV_8UC1))\
            for filename in in_path.iterdir()]
min_shape = None
min_n = 100000
min_m = 100000
for name, img in imgs:
    n, m = img.shape
    if n < min_n:
        min_n = n
    if m < min_m:
        min_m = m
min_shape = (min_n, min_m)

crop_shape = (110, 190)

cropped_imgs = [(filename, crop_image(img, crop_shape))
                    for filename, img in imgs]

side_lenght = 388
resized_imgs = [(filename, cv2.resize(img, dsize=None, fx=side_lenght / img.shape[0],
                                    fy=side_lenght / img.shape[0],
                                    interpolation=cv2.INTER_CUBIC))\
                    for filename, img in cropped_imgs]

if not out_path.exists():
    out_path.mkdir()
for img_name, img in resized_imgs:
    cv2.imwrite('{}/{}.png'.format(out_path.as_posix(), img_name), img)
