import cv2
import numpy as np

import colormotion.dataset as dataset


def write_image_lab(filename, l, ab):
    image = np.round(255 * np.clip(dataset.lab_to_bgr(l, ab), 0, 1)).astype('uint8')
    if not cv2.imwrite(filename, image):
        raise RuntimeError('Failed to write {}'.format(filename))

image = np.load('output_000214_77.npy')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.imshow(image[:, :, 0])
plt.figure(2)
plt.imshow(image[:, :, 1])
plt.figure(3)
plt.imshow(image[:, :, 2])
# plt.figure(4)
plt.show()

input()
fig = plt.figure()
ax1 = fig.add_subplot(500)
# Bilinear interpolation
ax1.imshow(image, cmap=cm.Greys_r)
ax2 = fig.add_subplot(500)
# 'nearest' interpolation
ax2.imshow(image, interpolation='nearest', cmap=cm.Greys_r)
plt.show()



for i in range(10):
    from pathlib import Path
    name = 'output_{}.png'.format(i)
    if not Path(name).exists():
        write_image_lab(str(name), image[:, :, 0:1], image[:, :, 1:])
        break
