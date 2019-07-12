import cv2

def crop(img, side):
    """Return cropped square image size of side x side given img numpy"""
    side = side // 2
    y = img.shape[0] // 2
    x = img.shape[1] // 2
    print(y, x)
    print(img.shape)
    return img[y - side:y +side, x - side : x+side]


if __name__ == "__main__":
    img = cv2.imread('../sample/dog2.jpeg')
    cv2.imshow('original', img)
    cropped = crop(img, 256)
    print(cropped.shape)
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
