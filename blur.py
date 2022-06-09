import math
import numpy as np
import cv2
from numba import jit


@jit
def cal_blur(imgarray, theta, delta, L, S=0):
    imgheight = imgarray.shape[0]
    imgwidth = imgarray.shape[1]
    c0 = int(imgheight / 2)
    c1 = int(imgwidth / 2)
    theta = theta / 180 * math.pi
    delta = delta / 180 * math.pi
    blurred_imgarray = np.copy(imgarray)
    for x in range(0, imgheight):
        for y in range(0, imgwidth):
            R = math.sqrt((x - c0) ** 2 + (y - c1) ** 2)
            alpha = math.atan2(y - c1, x - c0)
            X_cos = L * math.cos(delta) - S * R * math.cos(alpha)
            Y_sin = L * math.sin(delta) - S * R * math.sin(alpha)
            N = int(
                max(
                    abs(R * math.cos(alpha + theta) + X_cos + c0 - x),
                    abs(R * math.sin(alpha + theta) + Y_sin + c1 - y),
                )
            )
            if N <= 0:
                continue
            count = 0
            sum_r, sum_g, sum_b = 0, 0, 0
            for i in range(0, N + 1):
                n = i / N
                xt = int(R * math.cos(alpha + n * theta) + n * X_cos + c0)
                yt = int(R * math.sin(alpha + n * theta) + n * Y_sin + c1)
                if xt < 0 or xt >= imgheight:
                    continue
                elif yt < 0 or yt >= imgwidth:
                    continue
                else:
                    sum_r += imgarray[xt, yt][0]
                    sum_g += imgarray[xt, yt][1]
                    sum_b += imgarray[xt, yt][2]
                    count += 1
            blurred_imgarray[x, y] = np.array(
                [sum_r / count, sum_g / count, sum_b / count]
            )
    return blurred_imgarray

arr = 255 - cv2.imread(r"C:\Users\Administrator\Desktop\lane.png", cv2.IMREAD_UNCHANGED)[...,-1]
arr = arr.reshape([*arr.shape, 1])
arr = np.concatenate([arr]*3, axis=2)
empty = 255 * np.ones((200, 400, 3))
arr = np.concatenate([arr, empty], axis=1)
arr = cal_blur(arr, 20, 0, 0)[:, :-400]
cv2.imwrite(r"C:\Users\Administrator\Desktop\lane_b.png", arr.astype(np.uint8))
