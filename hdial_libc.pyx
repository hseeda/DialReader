import numpy as np
cimport numpy as np
from libc.math cimport round,sin,cos
import cv2

cpdef double test(np.ndarray img):
    cdef int  w = img.shape[0] // 2
    cdef double angr = 33.33
    cdef int i
    # for i in range(0,10):
    #     print("=========== ", i)
    return (angr)

cpdef double getDialAngle1(np.ndarray img, int steps):
    cdef int  w = img.shape[0] // 2
    cdef int  w_short = int(w * .8)
    cdef int  w_start = int(w * .3)
    cdef int  ang_res = 2
    cdef int  r_count = 10
    cdef int  r_start = w // 10
    cdef int  r_step = 2
    # --------------------------------------------------------------------------------
    cdef int p_old = 3*256 * 100
    cdef double ang_min = 0
    cdef double angr = 0
    cdef int x , y
    cdef int angle, p, p1, p2, p3, radius
    for angle in range(0, 360 * ang_res - 1):
        p = 0
        angr = angle / 180.0 * np.pi / ang_res
        for radius in range(w_start, w_short, r_step):
            x = int(round(w + radius * sin(angr)))
            y = int(round(w - radius * cos(angr)))
            p1 = img[y, x, 0]
            p2 = img[y, x, 1]
            p3 = img[y, x, 2]
            p = p + p1 + p2 + p3
        if p < p_old:
            p_old = p
            ang_min = angle
    ang_min_r = ang_min / 180.0 * np.pi /ang_res
    x = int(round(w + w * sin(ang_min_r)))
    y = int(round(w - w * cos(ang_min_r)))
    cv2.line(img, (w, w), (x, y), (0, 255, 0), 3)
    return  ang_min / ang_res
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cpdef tuple getOneCircle(np.ndarray img, double min_radius_ratio=0.25, double max_radius_ratio=0.75, \
                 double min_dist_between_centers=0.2):
    cdef np.ndarray g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # g_img = HED(img)
    # print("HED Called")

    cdef int w_img = int(img.shape[1])
    cdef int min_rs = int(w_img * min_radius_ratio)
    cdef int max_rs = int(w_img * max_radius_ratio)
    cdef int min_ds = int(w_img * min_dist_between_centers)
    if min_ds <= 0:
        min_ds = 1
    cdef int no_of_circles,x, y, r
    cdef np.ndarray circles

    (no_of_circles, circles) = getCirclesFromImage(g_img, min_rs, max_rs, min_ds)
    if no_of_circles == 0:
        return (0, 0, 0, 0)
    else:
        x = int(circles[0, 0, 0])
        y = int(circles[0, 0, 1])
        r = int(circles[0, 0, 2])
        return (x, y, r, no_of_circles)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cpdef tuple getCirclesFromImage(np.ndarray grey_img, int min_radius, int max_radius, int min_distance):
    cdef np.ndarray circles
    circles = cv2.HoughCircles( \
        grey_img, cv2.HOUGH_GRADIENT, \
        dp=1.0, \
        minDist=min_distance, \
        param1=15, \
        param2=50, \
        minRadius=min_radius, \
        maxRadius=max_radius)
    cdef int no_of_circles
    #    if circles is not None:
    if circles is None:
        return (0, 0)
    else:
        no_of_circles = circles.shape[1]
        return (no_of_circles, circles)

