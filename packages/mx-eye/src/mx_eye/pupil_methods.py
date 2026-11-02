"""Optional full-resolution pupil detectors; no training data or extra dependencies.

Starburst-style is an independent radial-edge implementation inspired by Li,
Winfield & Parkhurst (2005), not a reproduction of their complete algorithm.
All returned ellipse coordinates are local to the eye ROI.
"""
import math
import cv2
import numpy as np


def adaptive_mask(gray, settings):
    window = max(3, int(round(settings['pupil_adaptive_window'])) | 1)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, window,
                                 float(settings['pupil_adaptive_offset']))


def ellipse_coordinates(points, ellipse):
    (cx, cy), (width, height), angle = ellipse
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    delta = points - (cx, cy)
    u = (delta[:, 0]*cosine + delta[:, 1]*sine) / (width/2)
    v = (-delta[:, 0]*sine + delta[:, 1]*cosine) / (height/2)
    return u, v


def plausible(ellipse, shape, minimum, maximum):
    (cx, cy), (width, height), angle = ellipse
    if not np.isfinite([cx, cy, width, height, angle]).all():
        return False
    small, large = min(width, height), max(width, height)
    area = math.pi*width*height/4
    if small < 2 or large/small > 4 or not minimum <= area <= maximum:
        return False
    theta = math.radians(angle)
    rx = math.hypot(width*math.cos(theta), height*math.sin(theta))/2
    ry = math.hypot(width*math.sin(theta), height*math.cos(theta))/2
    h, w = shape
    return cx-rx >= -1 and cy-ry >= -1 and cx+rx <= w and cy+ry <= h


def residuals(points, ellipse):
    u, v = ellipse_coordinates(points, ellipse)
    # Approximate geometric distance to the ellipse, in source pixels.
    return np.abs(np.hypot(u, v)-1) * min(ellipse[1])/2


def coverage(points, ellipse):
    u, v = ellipse_coordinates(points, ellipse)
    bins = np.floor((np.arctan2(v, u)+np.pi)*12/(2*np.pi)).astype(int) % 12
    return np.unique(bins).size


def robust_ellipse(points, shape, minimum, maximum, tolerance):
    """Bounded deterministic consensus fit; reject unsupported/degenerate ellipses."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) < 8:
        return None
    if len(points) > 160:
        points = points[np.linspace(0, len(points)-1, 160).astype(int)]
    best, best_score = None, None
    rng = np.random.default_rng(0)
    for attempt in range(25):
        subset = points if attempt == 0 else points[rng.choice(len(points), 8, replace=False)]
        try:
            ellipse = cv2.fitEllipse(subset)
        except cv2.error:
            continue
        if not plausible(ellipse, shape, minimum, maximum):
            continue
        errors = residuals(points, ellipse)
        keep = errors <= tolerance
        count = int(keep.sum())
        if count < max(8, math.ceil(len(points)*.5)):
            continue
        spread = coverage(points[keep], ellipse)
        if spread < 6:
            continue
        score = (count, spread, -float(np.median(errors[keep])))
        if best_score is None or score > best_score:
            best, best_score = keep, score
        if count >= len(points)*.9 and spread >= 9:
            break
    if best is None:
        return None
    for _ in range(3):
        try:
            ellipse = cv2.fitEllipse(points[best])
        except cv2.error:
            return None
        if not plausible(ellipse, shape, minimum, maximum):
            return None
        keep = residuals(points, ellipse) <= tolerance
        if keep.sum() < max(8, math.ceil(len(points)*.5)) or coverage(points[keep], ellipse) < 6:
            return None
        if np.array_equal(keep, best):
            break
        best = keep
    return ellipse, points[best]


def sample(gray, points):
    points = np.asarray(points, np.float32)
    return cv2.remap(gray, points[:, 0].reshape(1, -1), points[:, 1].reshape(1, -1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


def boundary_contrast(gray, glare, ellipse):
    """Require a dark-inside/light-outside boundary away from reflections."""
    (cx, cy), (width, height), angle = ellipse
    phi = np.arange(48, dtype=np.float32)*(2*np.pi/48)
    theta = math.radians(angle)
    u, v = width/2*np.cos(phi), height/2*np.sin(phi)
    offsets = np.column_stack((u*math.cos(theta)-v*math.sin(theta),
                               u*math.sin(theta)+v*math.cos(theta)))
    inner, outer = (cx, cy)+.8*offsets, (cx, cy)+1.12*offsets
    h, w = gray.shape
    valid = ((outer[:, 0] >= 0) & (outer[:, 0] < w-1) &
             (outer[:, 1] >= 0) & (outer[:, 1] < h-1))
    valid &= (sample(glare, inner) == 0) & (sample(glare, outer) == 0)
    if valid.sum() < 12:
        return -float('inf')
    return float(np.median(sample(gray, outer[valid]).astype(float)-sample(gray, inner[valid])))


def candidate(ellipse, origin, label):
    (cx, cy), (width, height), _ = ellipse
    return dict(label=label, area=int(round(math.pi*width*height/4)),
                local_x=float(cx), local_y=float(cy), x=origin[0]+float(cx),
                y=origin[1]+float(cy), ellipse=ellipse,
                aspect=max(width, height)/min(width, height), fill=1.0)


def detect_edges(gray, settings, origin, seed=None):
    """Canny contours or radial rays, followed by robust ellipse validation."""
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    glare = cv2.dilate(cv2.compare(gray, int(round(settings['cr_thr'])), cv2.CMP_GT),
                       np.ones((3, 3), np.uint8))
    minimum, maximum = float(settings['pupil_min']), float(settings['pupil_max'])
    tolerance = float(settings['pupil_fit_error'])
    contrast = float(settings['pupil_edge_contrast'])
    candidates = []
    if settings['pupil_method'] == 'edge_ellipse':
        low = float(settings['pupil_edge_threshold'])
        evidence = cv2.Canny(smooth, low, low*2, L2gradient=True)
        evidence[glare != 0] = 0
        contours, _ = cv2.findContours(evidence, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Bound the more expensive fitting work on noisy/eyelash-heavy images.
        eligible = []
        for contour in contours:
            if len(contour) < 12:
                continue
            _, _, w, h = cv2.boundingRect(contour)
            if w*h < minimum or w*h > maximum*8:
                continue
            eligible.append(contour)
        for contour in sorted(eligible, key=len, reverse=True)[:16]:
            fit = robust_ellipse(contour, gray.shape, minimum, maximum, tolerance)
            if fit is None:
                continue
            ellipse, _ = fit
            if boundary_contrast(smooth, glare, ellipse) < contrast:
                continue
            if any(math.hypot(ellipse[0][0]-c['local_x'], ellipse[0][1]-c['local_y']) < 2 for c in candidates):
                continue
            candidates.append(candidate(ellipse, origin, len(candidates)+1))
        return candidates, evidence

    # Starburst-style: previous/manual center, ROI center, then three dark seeds.
    h, w = gray.shape
    seeds = []
    if seed is not None and 0 <= seed[0] < w and 0 <= seed[1] < h:
        seeds.append(tuple(seed))
    seeds.append((w/2, h/2))
    darkness = cv2.GaussianBlur(smooth, (9, 9), 0).copy()
    exclusion = max(4, int(math.sqrt(max(1, minimum)/math.pi)))
    for _ in range(3):
        _, _, location, _ = cv2.minMaxLoc(darkness)
        if all(math.hypot(location[0]-s[0], location[1]-s[1]) >= 3 for s in seeds):
            seeds.append(location)
        cv2.circle(darkness, location, exclusion, 255, -1)
    count = int(round(settings['pupil_rays']))
    angles = np.arange(count, dtype=np.float32)*(2*np.pi/count)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    # Up to aspect ratio 4: major semiaxis <= 2*sqrt(area/pi).
    radius = min(math.hypot(w, h), 2*math.sqrt(maximum/math.pi)+4)
    distances = np.arange(1, math.ceil(radius)+1, dtype=np.float32)
    evidence = np.zeros_like(gray)
    for initial in seeds:
        center = np.asarray(initial, np.float32)
        accepted = None
        for _ in range(3):
            coords = center + directions[:, None, :]*distances[None, :, None]
            mx, my = coords[:, :, 0], coords[:, :, 1]
            intensity = cv2.remap(smooth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            bright = cv2.remap(glare, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            inside = (mx >= 1) & (mx < w-1) & (my >= 1) & (my < h-1)
            good = inside[:, 2:] & inside[:, :-2] & (bright[:, 2:] == 0) & (bright[:, :-2] == 0)
            crossing = good & ((intensity[:, 2:]-intensity[:, :-2]) >= contrast)
            rays = np.flatnonzero(crossing.any(axis=1))
            if len(rays) < max(8, count//4):
                break
            first = crossing[rays].argmax(axis=1)+1
            points = coords[rays, first]
            fit = robust_ellipse(points, gray.shape, minimum, maximum, tolerance)
            if fit is None:
                break
            ellipse, inliers = fit
            if boundary_contrast(smooth, glare, ellipse) < contrast:
                break
            accepted = ellipse, inliers
            movement = np.linalg.norm(np.asarray(ellipse[0])-center)
            center = np.asarray(ellipse[0], np.float32)
            if movement < .5:
                break
        if accepted is not None:
            ellipse, inliers = accepted
            candidates.append(candidate(ellipse, origin, len(candidates)+1))
            pixels = np.rint(inliers).astype(int)
            evidence[np.clip(pixels[:, 1],0,h-1),np.clip(pixels[:, 0],0,w-1)] = 255
    return candidates, evidence
