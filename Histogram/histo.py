from matplotlib import pyplot as plt
import numpy as np
import cv2


def create_histogram(img):
    assert len(img.shape) == 2 
    histogram = [0] * 256
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            histogram[img[row, col]] += 1
    return histogram

def histogram_equalization(image):
    h = create_histogram(image)
    c = np.zeros_like(h)
    c[0] = h[0]
    for i in range(1, len(h)):
        c[i] = c[i-1] + h[i]

    L = 256
    N = image.size
    print(N)

    T = np.round((L - 1) * c / N).astype(np.uint8)
    print(len(T))
    equalized_image = T[image]
    print(equalized_image)
    return equalized_image

input_image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

equalized_image = histogram_equalization(input_image)
cv2.imshow('Original Image', input_image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.plot(create_histogram(input_image), color='b', label='Original Histogram')
plt.plot(create_histogram(equalized_image), color='r', label='Equalized Histogram')
plt.xlim([0, 256])
plt.title('Histogram for Grayscale Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.show()
