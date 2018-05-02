import cv2
import numpy as np

BBOX_PADDING = 100

def get_bounding_box(image):
    x_dim = np.where(image.max(0) == 255)[0]
    y_dim = np.where(image.max(1) == 255)[0]
    x_min, x_max = x_dim[0], x_dim[-1]
    y_min, y_max = y_dim[0], y_dim[-1]
    #ipdb.set_trace()
    return x_min, y_min, x_max, y_max

def crop_image(input_image):
    image = cv2.medianBlur(input_image, 3)
    s = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
    ret, s = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    
    img_y = image.shape[0]
    img_x = image.shape[1]
    X = int(img_x / 2)
    Y = int(img_y / 5)
    
    poly = np.array([[(0,0), (X,0), (0, Y)], [(X,0), (img_x, 0), (img_x, Y)]])
    s = cv2.fillPoly(s, poly, (0,0,0))
    
    kernel = np.ones((3,3), np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    x_min, y_min, x_max, y_max = get_bounding_box(s)
    x_min = max(x_min - BBOX_PADDING, 0)
    y_min = max(y_min - BBOX_PADDING, 0)
    x_max = min(x_max + BBOX_PADDING, img_x)
    y_max = min(y_max + BBOX_PADDING, img_y)
    
    cropped_image = input_image[y_min:y_max, x_min:x_max, :]
    return cropped_image

# edge = cv2.Canny(gray_image, 50, 100)
# plt.subplot(121), plt.imshow(edge, cmap='gray')
# plt.subplot(122), plt.imshow(gray_image, cmap='gray')
# plt.show()

# thresh = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
# plt.imshow(thresh, cmap='gray')
# plt.show()

# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel, iterations = 2)
# plt.imshow(opening, cmap='gray')
# plt.show()

# sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
# sure_fg = np.uint8(sure_fg)

# unknown = cv2.subtract(sure_bg,sure_fg)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    image_folder = "/home/max/data/robocup/tiny_athome_images"
    for file in os.listdir(image_folder):
        r = np.random.randint(len(os.listdir(image_folder)))
        image = cv2.imread(os.path.join(image_folder, file))
        cropped_image = crop_image(image)
        im = Image.fromarray(cropped_image)
        im.save("/home/max/cropped/" + file)
        plt.imshow(cropped_image)
        plt.show()
