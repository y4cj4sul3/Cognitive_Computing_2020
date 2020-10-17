import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# read image
img = Image.open('input.png')
# show input image
img.show()

# smoothing filter (box filter)
boxFilter = np.ones((7, 7), np.float32)/49
smoothed = cv2.filter2D(np.asarray(img), -1, boxFilter)
smoothed = Image.fromarray(smoothed)
# draw text
font = ImageFont.truetype('font/ArialCE.ttf', 12)
draw = ImageDraw.Draw(smoothed)
draw.text((10, 760), 'R08922070', font=font)
# save smoothed image
smoothed.save('smoothing.png')


# sharpening filter (img + laplacian filter)
sharpeningFilter = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])
sharpened = cv2.filter2D(np.asarray(img), -1, sharpeningFilter)
sharpened = Image.fromarray(sharpened)
# draw text
draw = ImageDraw.Draw(sharpened)
draw.text((10, 760), 'R08922070', font=font)
# save smoothed image
sharpened.save('sharpening.png')
