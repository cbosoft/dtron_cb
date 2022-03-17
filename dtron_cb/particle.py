import os

import cv2
import numpy as np
from imutils import perspective


class ParticleConstructionError(Exception):
    """Particle could not be constructed."""


def midp(pt1, pt2):
    return np.mean([pt1, pt2], axis=0)


def dist(pt1, pt2) -> float:
    return np.sum((pt2-pt1)**2)**0.5


def get_axes(box):
    """
    Get two axis of box (e.g. horiz. and vert.).

    Returns list of tuples of two points defining each axis.
    """
    tl, tr, br, bl = box
    t = midp(tl, tr)
    b = midp(bl, br)
    l = midp(tl, bl)
    r = midp(tr, br)
    return [(t, b), (l, r)]


def size_of_box(box):
    """
    Get size of rectange with corners defined by $box.

    Return (width, length).
    """
    axA, axB = get_axes(box)
    size = dist(*axA), dist(*axB)
    size = tuple(sorted(size))
    return size


class Particle:

    CSV_HEADER = ('image_file_name', 'width', 'length', 'aspect_ratio', 'area', 'perimeter', 'form_factor', 'circularity',
                  'convex_area', 'convex_perimeter', 'convexity',
                  'focus_GDER', 'confidence_score')

    def __init__(self, orig_img_fn: str, orig_image: np.ndarray, contour, px2um: float, conf_score: float):

        if len(contour) < 3:
            raise ParticleConstructionError('small contour')

        self.image_file_name = orig_img_fn
        self.contour = contour
        moments = cv2.moments(contour)

        self.conf_score = conf_score

        try:
            self.centroid = int(moments['m10']/moments['m00'])*px2um, int(moments['m01']/moments['m00'])*px2um
        except ZeroDivisionError:
            self.centroid = np.nan, np.nan

        self.area = cv2.contourArea(contour)*px2um*px2um
        self.perimeter = cv2.arcLength(contour, True)*px2um
        convex_hull = cv2.convexHull(contour)
        self.convex_area = cv2.contourArea(convex_hull)*px2um*px2um
        self.convex_perimeter = cv2.arcLength(convex_hull, True)*px2um
        try:
            self.solidity = self.area/self.convex_area
        except ZeroDivisionError:
            self.solidity = np.nan
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = perspective.order_points(box)
        self.width, self.length = size_of_box(box)
        self.width *= px2um
        self.length *= px2um
        self.aspect_ratio = self.width / self.length
        self.min_area_rect = np.int0(box)
        self.bbox = x, y, w, h = cv2.boundingRect(contour)  # x, y, w, h
        try:
            # BS ISO 9276-1:1998
            self.form_factor = 4*self.area*np.pi/self.perimeter**2
            self.circularity = np.sqrt(self.form_factor)
        except ZeroDivisionError:
            self.circularity = self.form_factor = np.nan

        try:
            self.convexity = self.convex_perimeter / self.perimeter
        except ZeroDivisionError:
            self.convexity = np.nan

        # TODO focus metrics
        cutout = orig_image[y:y+h, x:x+w]
        self.focus_GDER = self.fmeasure_GDER(cutout)

    def __lt__(self, other: "Particle") -> bool:
        return self.image_file_name < other.image_file_name

    @staticmethod
    def fmeasure_GDER(img: np.ndarray, w_size=15):
        # Create a Gaussian kernel
        N = w_size // 2
        sig = N / 2.5
        sig2 = sig * sig
        x = y = np.linspace(-N, N, w_size)
        x, y = np.meshgrid(x, y)
        g = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2)) / (2 * np.pi * sig)

        # Split into x and y kernels
        g_x = -x * g / sig2
        g_x = g_x / np.sum(np.abs(g_x))
        g_y = -y * g / sig2
        g_y = g_y / np.sum(np.abs(g_y))

        # x, y kernel convolutions
        r_x = cv2.filter2D(img, cv2.CV_8U, g_x)
        r_y = cv2.filter2D(img, cv2.CV_8U, g_y)
        fm = r_x ** 2 + r_y ** 2

        # Final value is the mean
        fm = fm.mean()
        return fm

    def to_dict(self) -> dict:
        return dict(
            image_file_name=str(self.image_file_name),
            width=float(self.width),
            length=float(self.length),
            aspect_ratio=float(self.aspect_ratio),
            bbox=[float(v) for v in self.bbox],
            min_area_rect=[(int(x), int(y)) for x, y in self.min_area_rect[:]],
            area=float(self.area),
            perimeter=float(self.perimeter),
            form_factor=float(self.form_factor),
            circularity=float(self.circularity),
            convex_area=float(self.convex_area),
            convex_perimeter=float(self.perimeter),
            convexity=float(self.convexity),
            centroid=[float(v) for v in self.centroid],
            focus_GDER=float(self.focus_GDER),
            confidence_score=float(self.conf_score)
        )

    @staticmethod
    def unit_of(k):
        if k in []:
            return 'px'
        elif k in ['length', 'width', 'perimeter', 'convex_perimeter']:
            return '$\\rm \\mu m$'
        elif k in ['area', 'convex_area']:
            return '$\\rm \\mu m^2$'
        else:
            return '-'

    @staticmethod
    def prep(v) -> str:
        if isinstance(v, str):
            if os.path.exists(v):
                v = os.path.normpath(v)
            return f'"{v}"'
        else:
            return f'{v}'

    def to_csv_line(self) -> str:
        d = self.to_dict()
        return ','.join([self.prep(d[k]) for k in self.CSV_HEADER])
