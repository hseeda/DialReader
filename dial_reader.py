from timeit import default_timer as timer
import hdial_lib as h  # including jit
import hdial_libc as hc  # cython
import cv2

CFLAG = 2  # % 1 jit 2 cython
if CFLAG == 1:
    print("--- using @jit")
elif CFLAG == 2:
    print("--- using cython")


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def do_it(img, w_img):
    cc = timer()
    if CFLAG == 1:
        xC, yC, radiusC, n11 = h.getOneCircle(img, .05, .95, .1)
    elif CFLAG == 2:
        xC, yC, radiusC, n11 = hc.getOneCircle(img, .05, .95, .1)
    else:
        xC, yC, radiusC, n11 = h.getOneCircle(img, .05, .95, .1)
    t = (- cc + timer()) * 1000
    print("Dial Reading    = [ %.3f \u03BCs ]" % t)
    h.drawCircle(img, xC, yC, radiusC, 2, 0, 255, 255)
    img_out = h.cropImageC(img, xC, yC, radiusC)
    img_out = h.imgDimScale(img_out, w_img)
    cc = timer()
    if CFLAG == 1:
        dial_reading = h.getDialAngle1(img_out, 10)
    elif CFLAG == 2:
        dial_reading = hc.getDialAngle1(img_out, 10)
    else:
        dial_reading = h.getDialAngle1(img_out, 10)
    t = (- cc + timer()) * 1000
    print("Dial Reading    = [ %.3f \u03BCs ]" % t)
    return img_out, dial_reading


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == "__main__":
    DEBUG = False
    # DEBUG = True
    w_img = 400
    dial_scale = 100
    for i in range(1, 17):
        fn = ("img/dial_%s.jpg" % i)
        print("File name       = {", fn, "}")
        s_img = cv2.imread(fn)
        s_img = h.imgDimScale(s_img, w_img)
        s_img_out, dial_reading_ = do_it(s_img, w_img)
        cv2.imshow("output", s_img_out)
        print("Dial Reading    = [ %.2f\u00B0 ]" % dial_reading_)
        print("press any key")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
