# interval_scanline_progressive_paint.py
# Интервальный построчный алгоритм — progressive: закрашивание фрагментов по строкам.
# Python 3.8+, PyQt5, numpy

import sys, math, random, time
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

CANVAS_W = 1100
CANVAS_H = 760

def look_at_matrix(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    C = np.array(camera_pos, dtype=float)
    T = np.array(target, dtype=float)
    UP = np.array(up, dtype=float)
    zc = T - C
    zn = np.linalg.norm(zc)
    if zn < 1e-9:
        zc = np.array([0.0, 0.0, 1.0])
    else:
        zc = zc / zn
    xc = np.cross(zc, UP)
    xn = np.linalg.norm(xc)
    if xn < 1e-9:
        xc = np.array([1.0, 0.0, 0.0])
    else:
        xc = xc / xn
    yc = np.cross(xc, zc)
    V = np.eye(4, dtype=float)
    V[0,0:3] = xc; V[1,0:3] = yc; V[2,0:3] = zc
    V[0:3,3] = -V[0:3,0:3] @ C
    return V

def project_with_camera_and_depth(pts4, camera_pos, target, up, focal):
    pts = np.array(pts4, dtype=float)
    V = look_at_matrix(np.array(camera_pos, dtype=float),
                       np.array(target, dtype=float),
                       np.array(up, dtype=float))
    pts_cam = (V @ pts.T).T
    res = []
    eps = 1e-9
    for p in pts_cam:
        x, y, z, w = p
        if z <= eps:
            res.append(None)
            continue
        t = focal / z
        res.append((x * t, y * t, z))
    return np.array(res, dtype=object), pts_cam

def make_convex_polygon_in_plane(center, normal, radius, n_verts):
    n = np.array(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
    angles = sorted([random.random() * 2*math.pi for _ in range(n_verts)])
    verts = []
    for ang in angles:
        r = radius * (0.6 + 0.4*random.random())
        p = np.array(center) + r * (math.cos(ang) * u + math.sin(ang) * v)
        verts.append((p[0], p[1], p[2], 1.0))
    return verts

def generate_demo_scene():
    scene = []
    rect_verts = [
        (1.0, 0.5, 1.0, 1.0),
        (1.0, 2.5, 1.0, 1.0),
        (2.5, 2.5, 1.0, 1.0),
        (2.5, 0.5, 1.0, 1.0),
    ]
    scene.append({'id': 0, 'verts': np.array(rect_verts, dtype=float), 'color': QtGui.QColor(70,130,180)})
    tri_verts = [
        (1.5, 1.5, 1.5, 1.0),
        (2.5, 2.5, 0.5, 1.0),
        (3.0, 1.0, 0.5, 1.0),
    ]
    scene.append({'id': 1, 'verts': np.array(tri_verts, dtype=float), 'color': QtGui.QColor(240,140,20)})
    return scene

def generate_random_scene(num_polys=6):
    scene = []
    for pid in range(num_polys):
        cx = random.uniform(-1.5, 1.5)
        cy = random.uniform(-1.0, 1.0)
        cz = random.uniform(-0.8, 1.2)
        center = (cx, cy, cz)
        nx = random.uniform(-0.8, 0.8)
        ny = random.uniform(-0.8, 0.8)
        nz = random.uniform(-0.6, 0.6)
        normal = (nx, ny, nz + 0.2)
        n_verts = random.randint(3, 6)
        radius = random.uniform(0.5, 1.4)
        verts = make_convex_polygon_in_plane(center, normal, radius, n_verts)
        color = QtGui.QColor.fromHsv(int((pid * 57) % 360), 200, 230)
        scene.append({'id': pid, 'verts': np.array(verts, dtype=float), 'color': color})
    return scene

def plane_from_points(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    a, b, c = n
    d = - (a*p0[0] + b*p0[1] + c*p0[2])
    return float(a), float(b), float(c), float(d)

class IntervalScanlineEngine(QtCore.QObject):
    updated = QtCore.pyqtSignal()
    logUpdated = QtCore.pyqtSignal(str)
    rowCompleted = QtCore.pyqtSignal(int)

    def __init__(self, scene, focal, width, height, scale_screen, camera_pos, camera_target, camera_up):
        super().__init__()
        self.scene = scene
        self.focal = float(focal)
        self.W = width; self.H = height
        self.scale_screen = scale_screen
        self.cx = width/2.0; self.cy = height/2.0
        self.camera_pos = np.array(camera_pos, dtype=float)
        self.camera_target = np.array(camera_target, dtype=float)
        self.camera_up = np.array(camera_up, dtype=float)

        self.projected = {}
        self.cam_coords = {}
        self.poly_planes = {}
        self._prepare_projection()

        self.cur_scanline = 0
        self.running = False
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.on_tick)
        self.timer.setInterval(20)

        self.current_intersections = []
        self.current_intervals = []
        self.scanline_zbuffer = np.full((self.W,), np.inf, dtype=float)

        self.frame_pid = np.full((self.H, self.W), -1, dtype=int)
        self.frame_z = np.full((self.H, self.W), np.inf, dtype=float)

        self.MIN_PIXEL_WIDTH = 1
        self.MAX_RECURSION_DEPTH = 20
        self.SCANLINE_TIME_LIMIT = 0.2

        self.finished = False

    def _log(self, s):
        self.logUpdated.emit(s)

    def set_camera(self, camera_pos):
        self.camera_pos = np.array(camera_pos, dtype=float)
        self._prepare_projection()

    def _prepare_projection(self):
        self.projected.clear(); self.cam_coords.clear(); self.poly_planes.clear()
        for p in self.scene:
            pid = p['id']
            proj, pts_cam = project_with_camera_and_depth(p['verts'], self.camera_pos, self.camera_target, self.camera_up, self.focal)
            self.projected[pid] = proj
            self.cam_coords[pid] = np.array([pc[:3] for pc in pts_cam], dtype=float)
            found = False
            cc = self.cam_coords[pid]; n = len(cc)
            for i in range(n):
                for j in range(i+1,n):
                    for k in range(j+1,n):
                        p0,p1,p2 = cc[i], cc[j], cc[k]
                        if np.linalg.norm(np.cross(p1-p0, p2-p0)) > 1e-8:
                            self.poly_planes[pid] = plane_from_points(p0,p1,p2)
                            found = True; break
                    if found: break
                if found: break
            if not found:
                self.poly_planes[pid] = None

    def reset(self):
        self.cur_scanline = 0
        self.current_intersections = []
        self.current_intervals = []
        self.scanline_zbuffer.fill(np.inf)
        self.frame_pid = np.full((self.H, self.W), -1, dtype=int)
        self.frame_z = np.full((self.H, self.W), np.inf, dtype=float)
        self._prepare_projection()
        self.finished = False
        self._log("Engine reset")
        self.updated.emit()

    def start(self):
        if not self.running:
            self.running = True
            self.timer.start()

    def pause(self):
        if self.running:
            self.running = False
            self.timer.stop()

    def on_tick(self):
        if self.cur_scanline >= self.H:
            if not self.finished:
                self.finished = True
                self._log("Finished all scanlines")
            self.pause()
            return
        start = time.time()
        try:
            self._process_scanline(self.cur_scanline, start)
            self.rowCompleted.emit(self.cur_scanline)
        except Exception as e:
            self._log(f"EXCEPTION during scanline {self.cur_scanline}: {e}")
        self.updated.emit()
        self.cur_scanline += 1

    def run_full(self):
        self.reset()
        for y in range(self.H):
            start = time.time()
            self._process_scanline(y, start)
            self.rowCompleted.emit(y)
            if (y & 31) == 0:
                QtWidgets.QApplication.processEvents()
        self.finished = True
        self.cur_scanline = self.H
        self._log("Finished all scanlines (run_full)")
        self.updated.emit()

    def z_at(self, pid, x_log, y_log):
        plane = self.poly_planes.get(pid, None)
        if plane is None: return None
        a,b,c,d = plane
        denom = a * x_log + b * y_log + c * self.focal
        if abs(denom) < 1e-12: return None
        s = - d / denom
        if s <= 0: return None
        z_cam = s * self.focal
        return float(z_cam)

    def intersection_x_of_planes(self, p1, p2, y_log):
        pl1 = self.poly_planes.get(p1, None); pl2 = self.poly_planes.get(p2, None)
        if pl1 is None or pl2 is None: return None
        a1,b1,c1,d1 = pl1; a2,b2,c2,d2 = pl2
        A = d1 * a2 - d2 * a1
        RHS = (d2 * b1 - d1 * b2) * y_log + (d2 * c1 - d1 * c2) * self.focal
        if abs(A) < 1e-12: return None
        x = RHS / A
        return float(x)

    def _process_scanline(self, y_px, start_time):
        if y_px < 0 or y_px >= self.H:
            return
        self.current_intersections.clear(); self.current_intervals.clear()
        self.scanline_zbuffer.fill(np.inf)
        y_log = (self.cy - (y_px + 0.5)) / self.scale_screen

        intersections = []
        for p in self.scene:
            pid = p['id']
            proj = self.projected.get(pid, None)
            if proj is None:
                proj, _ = project_with_camera_and_depth(p['verts'], self.camera_pos, self.camera_target, self.camera_up, self.focal)
            n = len(proj)
            for i in range(n):
                p1 = proj[i]; p2 = proj[(i+1)%n]
                if p1 is None or p2 is None: continue
                x1,y1,z1 = p1; x2,y2,z2 = p2
                if (y1 <= y_log and y_log <= y2) or (y2 <= y_log and y_log <= y1):
                    if abs(y2 - y1) < 1e-12:
                        continue
                    t = (y_log - y1) / (y2 - y1)
                    x_int = x1 + t * (x2 - x1)
                    x_px = int(round(self.cx + x_int * self.scale_screen))
                    intersections.append((pid, x_int, x_px))
        intersections.sort(key=lambda it: it[1])
        self.current_intersections = intersections.copy()

        if len(intersections) < 2:
            self._log(f"scanline {y_px}: no intersections")
            return

        active_now = set()
        for i in range(len(intersections) - 1):
            pid_i, x_i, px_i = intersections[i]
            if pid_i in active_now:
                active_now.remove(pid_i)
            else:
                active_now.add(pid_i)

            x_left = x_i
            x_right = intersections[i+1][1]
            if x_right <= x_left + 1e-12:
                continue

            xL_px = max(0, int(math.ceil(self.cx + x_left * self.scale_screen)))
            xR_px = min(self.W - 1, int(math.floor(self.cx + x_right * self.scale_screen)))
            if xR_px < xL_px:
                xR_px = xL_px

            if len(active_now) == 0:
                self.current_intervals.append((xL_px, xR_px, None))
            elif len(active_now) == 1:
                pid_single = next(iter(active_now))
                self.current_intervals.append((xL_px, xR_px, pid_single))
                for px in range(xL_px, xR_px + 1):
                    x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                    z = self.z_at(pid_single, x_log_px, y_log)
                    if z is not None:
                        if not np.isfinite(self.scanline_zbuffer[px]) or z < self.scanline_zbuffer[px]:
                            self.scanline_zbuffer[px] = z
                        if z < self.frame_z[y_px, px]:
                            self.frame_z[y_px, px] = z
                            self.frame_pid[y_px, px] = pid_single
            else:
                self._resolve_interval_stack(x_left, x_right, set(active_now), y_log, xL_px, xR_px, start_time, y_px)

        self._log(f"scanline {y_px}: intersections={len(intersections)} intervals={len(self.current_intervals)}")

    def _resolve_interval_stack(self, a_init, b_init, active_polys_set, y_log, xL_px_init, xR_px_init, start_time, y_px):
        stack = [(a_init, b_init, set(active_polys_set), 0)]
        ops = 0
        splits = 0
        seen_split = set()
        while stack:
            if (ops & 127) == 0:
                QtWidgets.QApplication.processEvents()
            ops += 1
            if time.time() - start_time > self.SCANLINE_TIME_LIMIT:
                while stack:
                    a,b,pols,depth = stack.pop()
                    pxL = max(0, int(math.ceil(self.cx + a * self.scale_screen)))
                    pxR = min(self.W - 1, int(math.floor(self.cx + b * self.scale_screen)))
                    if pxR < pxL: pxR = pxL
                    xm = 0.5*(a+b)
                    best_pid, best_z = None, np.inf
                    for pid in pols:
                        z = self.z_at(pid, xm, y_log)
                        if z is None: continue
                        if z < best_z:
                            best_z, best_pid = z, pid
                    self.current_intervals.append((pxL, pxR, best_pid))
                    if best_pid is not None:
                        for px in range(pxL, pxR+1):
                            x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                            z = self.z_at(best_pid, x_log_px, y_log)
                            if z is not None and z < self.frame_z[y_px, px]:
                                self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid
                self._log(f"scanline {y_px}: TIMEOUT bailout in [{a_init:.4f},{b_init:.4f}]")
                return

            a,b,pols,depth = stack.pop()
            pxL = max(0, int(math.ceil(self.cx + a * self.scale_screen)))
            pxR = min(self.W - 1, int(math.floor(self.cx + b * self.scale_screen)))
            if pxR < pxL: pxR = pxL
            pixel_width = max(1, pxR - pxL + 1)
            logical_width = b - a

            if pixel_width <= self.MIN_PIXEL_WIDTH or logical_width < 1e-6 or depth >= self.MAX_RECURSION_DEPTH:
                xm = 0.5*(a+b)
                best_pid, best_z = None, np.inf
                for pid in pols:
                    z = self.z_at(pid, xm, y_log)
                    if z is None: continue
                    if z < best_z:
                        best_z, best_pid = z, pid
                self.current_intervals.append((pxL, pxR, best_pid))
                if best_pid is not None:
                    for px in range(pxL, pxR+1):
                        x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                        z = self.z_at(best_pid, x_log_px, y_log)
                        if z is not None and z < self.frame_z[y_px, px]:
                            self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid
                continue

            if len(pols) == 0:
                self.current_intervals.append((pxL, pxR, None))
                continue
            if len(pols) == 1:
                pid = next(iter(pols))
                self.current_intervals.append((pxL, pxR, pid))
                for px in range(pxL, pxR+1):
                    x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                    z = self.z_at(pid, x_log_px, y_log)
                    if z is not None and z < self.frame_z[y_px, px]:
                        self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = pid
                continue

            candidate_xis = []
            pols_list = list(pols)
            for i in range(len(pols_list)):
                for j in range(i+1, len(pols_list)):
                    p1 = pols_list[i]; p2 = pols_list[j]
                    eps_inner = max(1e-9, 1e-9 * logical_width)
                    xL_test = a + eps_inner; xR_test = b - eps_inner
                    z1L = self.z_at(p1, xL_test, y_log); z2L = self.z_at(p2, xL_test, y_log)
                    z1R = self.z_at(p1, xR_test, y_log); z2R = self.z_at(p2, xR_test, y_log)
                    if z1L is None or z2L is None or z1R is None or z2R is None:
                        continue
                    sL = math.copysign(1.0, z1L - z2L) if abs(z1L - z2L) > 1e-12 else 0.0
                    sR = math.copysign(1.0, z1R - z2R) if abs(z1R - z2R) > 1e-12 else 0.0
                    if sL != sR:
                        xi = self.intersection_x_of_planes(p1, p2, y_log)
                        if xi is None: continue
                        tol = max(1e-9, 1e-9 * logical_width)
                        if xi <= a + tol or xi >= b - tol:
                            continue
                        candidate_xis.append((xi, p1, p2))
            if candidate_xis:
                candidate_xis.sort(key=lambda t: t[0])
                xi, pa, pb = candidate_xis[0]
                key = round(xi, 9)
                if key in seen_split or splits > 20000:
                    xm = 0.5*(a+b)
                    best_pid, best_z = None, np.inf
                    for pid in pols:
                        z = self.z_at(pid, xm, y_log)
                        if z is None: continue
                        if z < best_z:
                            best_z, best_pid = z, pid
                    self.current_intervals.append((pxL, pxR, best_pid))
                    if best_pid is not None:
                        for px in range(pxL, pxR+1):
                            x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                            z = self.z_at(best_pid, x_log_px, y_log)
                            if z is not None and z < self.frame_z[y_px, px]:
                                self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid
                    continue
                seen_split.add(key); splits += 1
                stack.append((xi, b, set(pols), depth+1))
                stack.append((a, xi, set(pols), depth+1))
                continue

            xm = 0.5*(a+b)
            best_pid, best_z = None, np.inf
            for pid in pols:
                z = self.z_at(pid, xm, y_log)
                if z is None: continue
                if z < best_z:
                    best_z, best_pid = z, pid
            self.current_intervals.append((pxL, pxR, best_pid))
            if best_pid is not None:
                for px in range(pxL, pxR+1):
                    x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                    z = self.z_at(best_pid, x_log_px, y_log)
                    if z is not None and z < self.frame_z[y_px, px]:
                        self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid

class GLWidget(QtWidgets.QWidget):
    def __init__(self, scene=None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(CANVAS_W, CANVAS_H)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.scene = scene if scene is not None else generate_demo_scene()
        self.camera_pos = np.array([0.0, 0.0, 15.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=float)
        self.focal = 4.0
        self.scale_screen = min(self.width(), self.height()) * 0.5

        self.engine = IntervalScanlineEngine(self.scene, self.focal, self.width(), self.height(), self.scale_screen, self.camera_pos, self.camera_target, self.camera_up)
        self.engine.logUpdated.connect(self.on_engine_log)
        self.engine.updated.connect(self.update)
        self.engine.rowCompleted.connect(self.on_engine_row_completed)

        self.poly_pen = QtGui.QPen(QtGui.QColor(40,40,160), 1)
        self.axis_pen_x = QtGui.QPen(QtGui.QColor(200,40,40), 2)
        self.axis_pen_y = QtGui.QPen(QtGui.QColor(40,160,40), 2)
        self.axis_pen_z = QtGui.QPen(QtGui.QColor(30,90,200), 2)

        self._recalc_camera_spherical()

        self.log_widget = None
        self.last_mouse = None
        self.mouse_mode = None

        self.frame_image = None
        self._init_frame_image()

        self.show_axes = True

    def _init_frame_image(self):
        w, h = max(1, self.width()), max(1, self.height())
        self.frame_image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        bg = QtGui.QColor(245,245,245).rgba()
        self.frame_image.fill(bg)

    def set_log_widget(self, widget):
        self.log_widget = widget; widget.clear()

    def on_engine_log(self, s):
        if self.log_widget is not None:
            self.log_widget.append(s)

    def _recalc_camera_spherical(self):
        v = self.camera_pos - self.camera_target
        r = np.linalg.norm(v)
        if r < 1e-9: r = 1e-9; v = np.array([r,0,0], dtype=float)
        self.cam_radius = r
        self.cam_el = math.asin(v[1] / r)
        self.cam_az = math.atan2(v[2], v[0])

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(245,245,245))
        w,h = self.width(), self.height()
        self.scale_screen = min(w,h) * 0.5
        self.engine.scale_screen = self.scale_screen
        if (self.engine.W != w) or (self.engine.H != h):
            self.engine.W = w; self.engine.H = h
            self.engine.cx = w/2.0; self.engine.cy = h/2.0
            self.engine.reset()
            self._init_frame_image()

        cx, cy = w/2.0, h/2.0

        final_polys_2d = []
        for p in self.scene:
            proj, pts_cam = project_with_camera_and_depth(p['verts'], self.camera_pos, self.camera_target, self.camera_up, self.focal)
            poly2d = []
            for v in proj:
                if v is None:
                    poly2d.append(None)
                else:
                    x_log, y_log, z_cam = v
                    sx = cx + x_log * self.scale_screen
                    sy = cy - y_log * self.scale_screen
                    poly2d.append((sx, sy))
            final_polys_2d.append((p['id'], poly2d, p['color']))

        # draw frame_image always if exists — this shows accumulated painted rows even during progressive run
        if self.frame_image is not None:
            qp.drawImage(0, 0, self.frame_image)

        # draw axes only if visible
        if self.show_axes:
            axis_len = 1.2
            origin = np.array([0.0,0.0,0.0,1.0])
            axis_points = [(origin, np.array([axis_len,0.0,0.0,1.0])),
                        (origin, np.array([0.0,axis_len,0.0,1.0])),
                        (origin, np.array([0.0,0.0,axis_len,1.0]))]
            pts = np.array([p for pair in axis_points for p in pair], dtype=float)
            proj_axes, _ = project_with_camera_and_depth(pts, self.camera_pos, self.camera_target, self.camera_up, self.focal)
            for idx, pen in enumerate((self.axis_pen_x, self.axis_pen_y, self.axis_pen_z)):
                p0 = proj_axes[2*idx]; p1 = proj_axes[2*idx+1]
                if p0 is None or p1 is None: continue
                sx1, sy1 = cx + p0[0]*self.scale_screen, cy - p0[1]*self.scale_screen
                sx2, sy2 = cx + p1[0]*self.scale_screen, cy - p1[1]*self.scale_screen
                qp.setPen(pen); qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

        # draw polygon outlines on top
        qp.setPen(self.poly_pen)
        for pid, poly2d, color in final_polys_2d:
            n = len(poly2d)
            for i in range(n):
                p1 = poly2d[i]; p2 = poly2d[(i+1)%n]
                if p1 is None or p2 is None: continue
                qp.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))

        # progressive indicators: scanline, intersections, intervals
        y_scan = self.engine.cur_scanline
        if 0 <= y_scan < h:
            qp.setPen(QtGui.QPen(QtGui.QColor(200,20,20), 1, QtCore.Qt.DashLine))
            qp.drawLine(0, y_scan, w, y_scan)
            qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10), 1))
            for (pid, x_log, x_px) in self.engine.current_intersections:
                qp.setBrush(QtGui.QColor(255,255,255))
                qp.drawEllipse(QtCore.QPointF(x_px, y_scan), 3, 3)
                qp.drawText(x_px+4, y_scan-4, f"{pid}")
            for (xL_px, xR_px, pid) in self.engine.current_intervals:
                rectL = max(0, xL_px); rectR = min(w-1, xR_px)
                if rectL > rectR: continue
                wseg = rectR - rectL + 1
                if pid is None:
                    brush = QtGui.QBrush(QtGui.QColor(200,200,200,80))
                else:
                    color = None
                    for p in self.scene:
                        if p['id'] == pid:
                            color = p['color']; break
                    if color is None:
                        brush = QtGui.QBrush(QtGui.QColor(200,200,200,80))
                    else:
                        c = QtGui.QColor(color); c.setAlpha(150); brush = QtGui.QBrush(c)
                qp.setBrush(brush); qp.setPen(QtGui.QPen(QtGui.QColor(80,80,80), 0))
                qp.drawRect(rectL, y_scan-6, wseg, 12)

        qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10))); qp.setFont(QtGui.QFont("Consolas", 11))
        run_state = "Выполняется" if self.engine.running else "Остановлен"
        qp.drawText(10, 20, f"Интервальный алгоритм — {run_state}    Строка: {self.engine.cur_scanline}/{self.height()}")

    def on_engine_row_completed(self, y):
        """
        When a row completes we copy engine.frame_pid for that row into frame_image.
        Also — to make progressive painting immediate and visible — paint current_intervals
        (which the engine computed for this row) directly into the image for that row.
        """
        try:
            if self.frame_image is None:
                return
            img_w = self.frame_image.width()
            img_h = self.frame_image.height()
            if y < 0 or y >= img_h:
                return

            pid_to_rgba = {}
            for p in self.scene:
                c = p['color']
                pid_to_rgba[p['id']] = QtGui.qRgba(c.red(), c.green(), c.blue(), 255)
            bg = QtGui.QColor(245,245,245).rgba()

            # First: copy the engine.frame_pid row (this is the authoritative z-tested per-pixel assignment)
            arr_pid = self.engine.frame_pid
            max_x = min(img_w, arr_pid.shape[1])
            for x in range(max_x):
                pid = int(arr_pid[y, x])
                if pid >= 0:
                    self.frame_image.setPixel(x, y, pid_to_rgba.get(pid, bg))
                else:
                    self.frame_image.setPixel(x, y, bg)

            # Second: ensure the current_intervals (as computed for this row) are also rendered
            # (this helps when intervals are coarse/larger than per-pixel updates or when frame_pid lags)
            for (xL_px, xR_px, pid) in self.engine.current_intervals:
                if pid is None: continue
                xL = max(0, xL_px); xR = min(img_w-1, xR_px)
                if xL > xR: continue
                color_rgba = pid_to_rgba.get(pid, bg)
                for x in range(xL, xR+1):
                    self.frame_image.setPixel(x, y, color_rgba)

            # update only the small stripe — cheap repaint
            self.update(0, y, max_x, 2)
        except Exception as ex:
            self.on_engine_log(f"Exception in on_engine_row_completed(y={y}): {ex}")

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.show_axes = True
            if self.engine.finished:
                self.engine.finished = False
                self.engine.frame_pid.fill(-1)
                self.engine.frame_z.fill(np.inf)
                self._init_frame_image()
            self.mouse_mode = "orbit_cam"
            v = self.camera_pos - self.camera_target
            r = np.linalg.norm(v)
            if r < 1e-9: r = 1e-9; v = np.array([r,0,0])
            self._orbit_start = (r, math.asin(v[1]/r), math.atan2(v[2], v[0]))
            self._orbit_start_mouse = ev.pos()
        self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_mouse is None: return
        pos = ev.pos()
        if getattr(self, "mouse_mode", None) == "orbit_cam":
            start_r, start_el, start_az = self._orbit_start
            start_mouse = self._orbit_start_mouse
            ddx = pos.x() - start_mouse.x(); ddy = pos.y() - start_mouse.y()
            az = start_az + ddx * 0.01
            el = start_el + ddy * 0.01
            max_el = math.radians(89.0)
            el = max(-max_el, min(max_el, el))
            r = start_r
            x = r * math.cos(el) * math.cos(az)
            y = r * math.sin(el)
            z = r * math.cos(el) * math.sin(az)
            self.camera_pos = np.array([x, y, z], dtype=float) + self.camera_target
            self.engine.set_camera(self.camera_pos)
            self._recalc_camera_spherical()
            self.update()
        self.last_mouse = pos

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None
        self.last_mouse = None

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            self.reset_camera()
        elif k == QtCore.Qt.Key_Space:
            self.start_instant()
        elif k == QtCore.Qt.Key_S:
            self.engine.on_tick()
        self.update()

    def reset_camera(self):
        self.camera_pos = np.array([0.0, 0.0, 15.0], dtype=float)
        self._recalc_camera_spherical()
        self.engine.set_camera(self.camera_pos)
        self.engine.reset()
        self._init_frame_image()
        self.show_axes = True
        self.update()

    def start_progressive(self):
        self.engine.reset()
        self.show_axes = False
        self.engine.start()
        self.update()

    def stop_progressive(self):
        self.engine.pause()
        self.update()

    def start_instant(self):
        self.engine.reset()
        self.show_axes = False
        self.engine.run_full()
        self.update()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интервальный построчный алгоритм — progressive paint")
        self.gl = GLWidget()
        self.setCentralWidget(self.gl)
        self.create_controls()
        self.resize(1300, 780)

    def create_controls(self):
        panel = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel("<b>Управление алгоритмом</b>"))

        btn_instant = QtWidgets.QPushButton("Run instant (Space)")
        btn_instant.clicked.connect(lambda: self.gl.start_instant())
        btn_progressive = QtWidgets.QPushButton("Start Progressive")
        btn_progressive.clicked.connect(lambda: self.gl.start_progressive())
        btn_stop = QtWidgets.QPushButton("Stop Progressive")
        btn_stop.clicked.connect(lambda: self.gl.stop_progressive())
        btn_step = QtWidgets.QPushButton("Шаг — 1 строка (S)")
        btn_step.clicked.connect(lambda: self.gl.engine.on_tick())
        btn_reset_cam = QtWidgets.QPushButton("Сброс камеры (R)")
        btn_reset_cam.clicked.connect(lambda: (self.gl.reset_camera(), self.gl.update()))
        layout.addWidget(btn_instant); layout.addWidget(btn_progressive); layout.addWidget(btn_stop)
        layout.addWidget(btn_step); layout.addWidget(btn_reset_cam)

        layout.addWidget(QtWidgets.QLabel("<b>Сцены</b>"))
        btn_demo = QtWidgets.QPushButton("Сгенерировать демо сцену (rect+triangle)")
        btn_demo.clicked.connect(self.on_demo_scene)
        btn_random = QtWidgets.QPushButton("Сгенерировать случайную сцену")
        btn_random.clicked.connect(self.on_random_scene)
        layout.addWidget(btn_demo); layout.addWidget(btn_random)

        layout.addWidget(QtWidgets.QLabel("<b>Журнал</b>"))
        self.log_view = QtWidgets.QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setFixedWidth(360)
        layout.addWidget(self.log_view)
        self.gl.set_log_widget(self.log_view)
        self.gl.engine.logUpdated.connect(lambda s: self.log_view.append(s))

        instr = QtWidgets.QLabel(
            "- ЛКМ + перетаскивание: вращение камеры (орбита вокруг цели). При начале вращения оси возвращаются и composited-frame очищается.\n"
            "- Run instant: выполнить полную отрисовку (instant)\n"
            "- Start Progressive: запустить анимированный прогон построчно (видно, как строки закрашиваются)\n"
            "- Stop Progressive: приостановить progressive\n"
            "- S: выполнить 1 строку вручную\n"
            "- R: сброс камеры и движка\n"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)
        layout.addStretch()
        dock = QtWidgets.QDockWidget("Панель управления", self)
        panel.setLayout(layout); dock.setWidget(panel); self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def on_demo_scene(self):
        self.gl.scene = generate_demo_scene()
        self.gl.engine.scene = self.gl.scene
        self.gl.engine.reset()
        self.gl._init_frame_image()
        self.gl.show_axes = True
        self.gl.update()

    def on_random_scene(self):
        self.gl.scene = generate_random_scene(random.randint(5,6))
        self.gl.engine.scene = self.gl.scene
        self.gl.engine.reset()
        self.gl._init_frame_image()
        self.gl.show_axes = True
        self.gl.update()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()