import numpy as np
from numba import jit, njit
import cv2
import warnings

warnings.filterwarnings('ignore')


# %% show image using cv2
def show(img):
    cv2.imshow("img", img);
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %% get npArray dims
def npd(in1):
    print('shape  : ', in1.shape)


# %% get npArray attributes
def npa(in1):
    print('type   : ', type(in1))
    print('ndim   : ', in1.ndim)
    print('shape  : ', in1.shape)
    print('size   : ', in1.size)
    print('dtype  : ', in1.dtype)
    print('strides: ', in1.strides)
    print('-----------------------')


# %% get npArray Statistics
def nps(in1):
    print('min     : ', in1.min())
    print('max     : ', in1.max())
    print('mean    : ', in1.mean())
    print('std     : ', in1.std())
    print('variance: ', np.var(in1))
    print('m25%    : ', np.percentile(in1, 25))
    print('m50%    : ', np.percentile(in1, 50))
    print('m75%    : ', np.percentile(in1, 75))
    print('-----------------------')


# %% convert to float32
def toFloat(in1):
    return in1.astype(np.float32)


# %% convert to float32
def toInt(in1):
    return in1.astype(np.uint8)


# %% scale input img
def imgScale(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


# %% scale input img by dimensions
def imgDimScale(img, i_width, i_height=0):
    if i_height == 0:
        scale = float(i_width) / float(img.shape[1])
    else:
        scale1 = float(i_width) / float(img.shape[1])
        scale2 = float(i_height) / float(img.shape[0])
        if scale1 > scale2:
            scale = scale1
        else:
            scale = scale2
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    #    print(scale,width,height)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


@jit
def getCirclesFromImage(grey_img, min_radius, max_radius, min_distance):
    circles = cv2.HoughCircles(
        grey_img, cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_distance,
        param1=15,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius)
    #    if circles is not None:
    if circles is None:
        return 0, 0
    else:
        no_of_circles = circles.shape[1]
        return no_of_circles, circles


# %% get one circle from color image
@jit
def getOneCircle(img, min_radius_ratio=0.25, max_radius_ratio=0.75, \
                 min_dist_between_centers=0.2):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # g_img = HED(img)
    # print("HED Called")

    w_img = int(img.shape[1])
    min_rs = int(w_img * min_radius_ratio)
    max_rs = int(w_img * max_radius_ratio)
    min_ds = int(w_img * min_dist_between_centers)
    if min_ds <= 0:
        min_ds = 1
    (no_of_circles, circles) = getCirclesFromImage(g_img, min_rs, max_rs, min_ds)
    if no_of_circles == 0:
        return 0, 0, 0, 0
    else:
        x = int(circles[0, 0, 0])
        y = int(circles[0, 0, 1])
        r = int(circles[0, 0, 2])
        return x, y, r, no_of_circles


# %% get one circle from color image
def getAverageCircle(img, min_radius_ratio=0.25, max_radius_ratio=0.75, \
                     min_dist_between_centers=0.2):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w_img = int(img.shape[1])
    min_rs = int(w_img * min_radius_ratio)
    max_rs = int(w_img * max_radius_ratio)
    min_ds = int(w_img * min_dist_between_centers)
    if min_ds <= 0:
        min_ds = 1
    (no_of_circles, circles) = getCirclesFromImage(g_img, min_rs, max_rs, min_ds)
    if no_of_circles == 0:
        return 0, 0, 0, 0
    else:
        x = int(np.average(circles[0, :, 0]))
        y = int(np.average(circles[0, :, 1]))
        r = int(np.average(circles[0, :, 2]))
        return x, y, r, no_of_circles


# %% get circles from color image and draw them all on out_img
def imgCircles(img, min_radius_ratio=0.25, max_radius_ratio=0.75, \
               min_dist_between_centers=0.2):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = img.copy()
    w_img = int(img.shape[1])
    min_rs = int(w_img * min_radius_ratio)
    max_rs = int(w_img * max_radius_ratio)
    min_ds = int(w_img * min_dist_between_centers)
    if min_ds <= 0:
        min_ds = 1
    (no_of_circles, circles) = getCirclesFromImage(g_img, min_rs, max_rs, min_ds)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        #        npa(circles)
        #        nps(circles)
        #        print(circles)
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(out_img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(out_img, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
    return no_of_circles, out_img


# %% draw circle on color image
def drawCircle(img, x, y, r, line_width=2, cR=0, cG=255, cB=0):
    cv2.circle(img, (x, y), r, (cR, cG, cB), line_width)
    cv2.rectangle(img, (x - line_width, y - line_width),
                  (x + line_width, y + line_width), (cR // 2, cG // 2, cB // 2), -1)


# %% crop image
def cropImage(img, x_start, x_end, y_start, y_end):
    width = img.shape[1]
    height = img.shape[0]
    xs = int(width * x_start)
    xe = int(width * x_end)
    ys = int(height * y_start)
    ye = int(height * y_end)
    return img[ys:ye, xs:xe, :]


# %% crop image
def cropImageC(img, x, y, radius, border=2):
    xx = int(x - radius - border)
    yy = int(y - radius - border)
    ww = int(2 * (radius + border))
    return img[yy:yy + ww, xx:xx + ww, :]


# %% get line from color image
def getOneLine(img):
    low_threshold = 20
    high_threshold = 220
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # HoughLines(image,  rho, theta, threshold)
    # image     An object of the class Mat representing the source (input) image.
    # rho       A variable of the type double representing the
    #             resolution of the parameter r in pixels.
    # theta     A variable of the type double representing the resolution of
    #             the parameter Φ in radians.
    # threshold A variable of the type integer representing the minimum number
    #             of intersections to “detect” a line.

    lines = cv2.HoughLines(edges, 1, np.pi / 360, 5)

    for rho1, theta1 in lines[0]:
        a = np.cos(theta1)
        b = np.sin(theta1)
        x0 = a * rho1
        y0 = b * rho1
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return x1, y1, x2, y2
    return 0, 0, 0, 0


# %% get line from color image
def getOneLineP(img, xc, yc):
    low_threshold = 50
    high_threshold = 200
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 360  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), \
                            min_line_length, max_line_gap)

    # print("xc = {0} , yc = {1}".format(xc, yc))
    # find line closest to image center
    dold = 9999
    di = 0
    i = 0
    try:
        imgc = img.copy()
        for line in lines:
            x1 = line[0, 0]
            y1 = line[0, 1]
            x2 = line[0, 2]
            y2 = line[0, 3]
            p1 = np.array([xc, yc])
            p2 = np.array([x1, y1])
            p3 = np.array([x2, y2])
            d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            d1 = pow(pow(xc - x1, 2) + pow(yc - y1, 2), .5)
            d2 = pow(pow(xc - x2, 2) + pow(yc - y2, 2), .5)
            d = d + min(d1, d2)
            if dold > d:
                dold = d
                di = i
            # print("d = {0} , dold = {1} , di = {2}".format(d, dold, di))
            # print("%s , %s, %s, %s" % (x1, y1, x2, y2))
            cv2.line(imgc, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0, 2], lines[i, 0, 3]), (0, 10 * i, 0), 4)
            i = i + 1
            # show(imgc)
        # print(lines[di, 0, 0], lines[di, 0, 1], lines[di, 0, 2], lines[di, 0, 3])
        cv2.line(imgc, (lines[di, 0, 0], lines[di, 0, 1]), (lines[di, 0, 2], lines[di, 0, 3]), (255, 0, 255), 4)
        # show(imgc)
        return lines[di, 0, 0], lines[di, 0, 1], lines[di, 0, 2], lines[di, 0, 3], dold
    except:
        print("skipped---------------")
        return 0, 0, 1, 1, 9999


# %% get angle
def getAngleNorth(x1, y1, x2, y2):
    return np.arctan((x2 - x1) / (y2 - y1)) * 180 / np.pi


# %%
def angleToXY_North(x_center, y_center, radius, angle):
    angle = angle - 90
    x9 = int(x_center + (radius * np.cos(angle * np.pi / 180)))
    y9 = int(y_center + (radius * np.sin(angle * np.pi / 180)))
    return x9, y9


class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0;
        self.startY = 0;
        self.endX = 0;
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W;
        self.endY = self.startY + H
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]


@jit(fastmath=True)
def drawDial(img, steps):
    w = img.shape[0] // 2

    for i in range(0, steps):
        a = i / steps * 360 / 180 * np.pi

        x = int(round(w + w * .8 * np.sin(a), 0))
        y = int(round(w - w * .8 * np.cos(a), 0))

        x1 = int(round(w + w * .98 * np.sin(a), 0))
        y1 = int(round(w - w * .98 * np.cos(a), 0))

        cv2.line(img, (x, y), (x1, y1), (0, 0, 255), 2)


@jit(fastmath=True)
def getDialAngle1(img, steps):
    w = img.shape[0] // 2
    w_short = int(w * .8)
    w_start = int(w * .3)
    ang_res = 2
    r_count = 10
    r_start = w // 10
    r_step = 2
    p_old = 3 * 256 * 100
    ang_min = 0
    for angle in range(0, 360 * ang_res - 1):
        p = 0
        angr = angle / 180.0 * np.pi / ang_res
        for radius in range(w_start, w_short, r_step):
            x = int(round(w + radius * np.sin(angr), 0))
            y = int(round(w - radius * np.cos(angr), 0))
            # cv2.line(img, (w, w), (x, y), (0, 0, 255), 1)
            p1 = img[y, x, 0]
            # p1 = img[y, x]
            p2 = img[y, x, 1]
            p3 = img[y, x, 2]
            p = p + p1 + p2 + p3
            # p = p + p1
            # img[y, x, 0] = 0
            # img[y, x, 1] = 0
            # img[y, x, 2] = 255
        # print(angle, p)
        if p < p_old:
            p_old = p
            ang_min = angle

    # print(ang_min)
    ang_min_r = ang_min / 180.0 * np.pi / ang_res
    x = int(round(w + w * np.sin(ang_min_r), 0))
    y = int(round(w - w * np.cos(ang_min_r), 0))
    cv2.line(img, (w, w), (x, y), (0, 255, 0), 3)
    drawDial(img, steps)
    return ang_min / ang_res
