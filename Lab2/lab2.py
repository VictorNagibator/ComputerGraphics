import sys, math, random, time
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

CANVAS_W = 1100
CANVAS_H = 760

# Матрица перевода в систему камеры (формирует преобразование в систему координат камеры)
# (аналогично предыдущей работе)
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

# Перспективная проекция (с учётом положения камеры и глубины (z по камере))
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

# Создать случайный многоугольник в заданной плоскости
def make_polygon(center, normal, radius, n_verts):
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

# Демонстрационная сцена (из лекции): прямоугольник + протыкающий треугольник
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

# Генерация случайной сцены с заданным количеством полигонов
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
        verts = make_polygon(center, normal, radius, n_verts)
        color = QtGui.QColor.fromHsv(int((pid * 57) % 360), 200, 230)
        scene.append({'id': pid, 'verts': np.array(verts, dtype=float), 'color': color})
    return scene

# Построение уравнения плоскости по трём точкам (ax + by + cz + d = 0)
def plane_from_points(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    a, b, c = n
    d = - (a*p0[0] + b*p0[1] + c*p0[2])
    return float(a), float(b), float(c), float(d)

# Класс интервального построчного сканирования
class IntervalScanline(QtCore.QObject):
    updated = QtCore.pyqtSignal()
    rowCompleted = QtCore.pyqtSignal(int)

    def __init__(self, scene, focal, width, height, scale_screen, camera_pos, camera_target, camera_up):
        super().__init__()
        # сцена: список полигонов (каждый полигон - словарь с 'id','verts','color')
        self.scene = scene
        # фокусное расстояние камеры в логических единицах
        self.focal = float(focal)
        # размеры экрана в пикселях
        self.W = width; self.H = height
        # масштаб логических координат в экранные пиксели
        self.scale_screen = scale_screen
        # координаты центра экрана в пикселях
        self.cx = width/2.0; self.cy = height/2.0
        # параметры камеры в мировых координатах
        self.camera_pos = np.array(camera_pos, dtype=float)
        self.camera_target = np.array(camera_target, dtype=float)
        self.camera_up = np.array(camera_up, dtype=float)

        # словари для хранения предварительно вычисленных результатов проекции
        # projected - список проекций вершин полигона в логических координатах (x_log, y_log, z_cam) или None
        # cam_coords - координаты вершин в системе камеры (для вычисления плоскости)
        # poly_planes - параметры плоскости (a,b,c,d) для каждого полигона или None
        self.projected = {}
        self.cam_coords = {}
        self.poly_planes = {}
        self._prepare_projection()

        # текущее положение сканирующей строки (в пикселях)
        self.cur_scanline = 0
        # флаг выполнения анимации
        self.running = False
        # таймер для пошаговой обработки строк в прогрессивном режиме
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.on_tick)
        self.timer.setInterval(20)

        # результат обработки текущей строки: пересечения и интервалы
        # current_intersections отсортирован по x_log
        # current_intervals список пиксельных интервалов, покрывающих строку
        self.current_intersections = []
        self.current_intervals = []

        # временный буфер глубин для текущей строки (размер = ширина экрана)
        # для каждого пикселя этой строки хранится минимальная найденная глубина (меньше -> ближе)
        self.scanline_zbuffer = np.full((self.W,), np.inf, dtype=float)

        # полный кадровый буфер: для каждого пикселя храним id полигона (или -1 для фона) и глубину
        # frame_pid[y,x] - id полигона, который в настоящий момент считается видимым в пикселе
        # frame_z[y,x] - глубина этого полигона (в системе камеры)
        self.frame_pid = np.full((self.H, self.W), -1, dtype=int)
        self.frame_z = np.full((self.H, self.W), np.inf, dtype=float)

        self.MIN_PIXEL_WIDTH = 1 # минимальная ширина в пикселях, на которой прекращаем дальнейшее разбиение
        self.MAX_RECURSION_DEPTH = 20 # максимальная глубина разбиения (иначе возможно переполнение стека из-за рекурсии)
        self.SCANLINE_TIME_LIMIT = 0.2 # максимальное время (в секундах) для обработки одной строки

        # флаг завершения полной отрисовки
        self.finished = False

    # пересчитать проекции и параметры плоскостей при изменении камеры или сцены
    def _prepare_projection(self):
        # очищаем старые данные
        self.projected.clear(); self.cam_coords.clear(); self.poly_planes.clear()
        # для каждого полигона вычисляем проекцию вершин и параметры плоскости в координатах камеры
        for p in self.scene:
            pid = p['id']
            proj, pts_cam = project_with_camera_and_depth(p['verts'], self.camera_pos, self.camera_target, self.camera_up, self.focal)
            # proj содержит кортежи (x_log, y_log, z_cam) или None для точек за камерой
            self.projected[pid] = proj
            # cam_coords - только трехмерные координаты в системе камеры (x,y,z)
            self.cam_coords[pid] = np.array([pc[:3] for pc in pts_cam], dtype=float)
            found = False
            cc = self.cam_coords[pid]; n = len(cc)

            # ищем три точки, чтобы определить плоскость полигона
            for i in range(n):
                for j in range(i+1,n):
                    for k in range(j+1,n):
                        p0,p1,p2 = cc[i], cc[j], cc[k]
                        # если векторы не коллинеарны, норма вектора ненулевая -> можно вычислить плоскость
                        if np.linalg.norm(np.cross(p1-p0, p2-p0)) > 1e-8:
                            self.poly_planes[pid] = plane_from_points(p0,p1,p2)
                            found = True; break
                    if found: break
                if found: break
            if not found:
                # вырожденный полигон или все точки на одной прямой -> плоскость не определена
                self.poly_planes[pid] = None

    # смена позиции камеры (и пересчёт проекций)
    def set_camera(self, camera_pos):
        self.camera_pos = np.array(camera_pos, dtype=float)
        # при смене камеры нужно пересчитать все проекции и плоскости
        self._prepare_projection()

    # сброс состояния алгоритма (очистка всех буферов и возврат к начальной строке)
    def reset(self):
        self.cur_scanline = 0
        self.current_intersections = []
        self.current_intervals = []
        # очищаем временный буфер строки
        self.scanline_zbuffer.fill(np.inf)
        # очищаем полный кадровый буфер
        self.frame_pid = np.full((self.H, self.W), -1, dtype=int)
        self.frame_z = np.full((self.H, self.W), np.inf, dtype=float)
        # пересчитываем проекции на случай, если сцена изменилась
        self._prepare_projection()
        # сбрасываем флаг завершения
        self.finished = False
        # сигнал об обновлении состояния (для перерисовки виджета)
        self.updated.emit()

    # старт прогрессивного режима (анимация построчной отрисовки)
    def start(self):
        if self.finished:
            # если уже отрисовано целиком, сначала делаем reset, чтобы запустить заново
            self.reset()
        if not self.running:
            self.running = True
            self.timer.start()

    # пауза прогрессивного режима
    def pause(self):
        if self.running:
            self.running = False
            self.timer.stop()

    # обработка одного шага таймера: обрабатываем текущую строку
    def on_tick(self):
        # если дошли до последней строки, помечаем завершение и останавливаем таймер
        if self.cur_scanline >= self.H:
            if not self.finished:
                self.finished = True
            self.pause()
            self.updated.emit()
            return
        start = time.time()

        # основной рабочий метод для одной строки
        self._process_scanline(self.cur_scanline, start)
        # уведомляем, что строка завершена
        self.rowCompleted.emit(self.cur_scanline)

        # сигнал об обновлении для перерисовки
        self.updated.emit()
        # переходим к следующей строке
        self.cur_scanline += 1

    # мгновенная отрисовка всего кадра (без анимации): просто обрабатываем все строки подряд
    def run_full(self):
        self.reset()
        for y in range(self.H):
            start = time.time()
            self._process_scanline(y, start)
            self.rowCompleted.emit(y)
            # чтобы окно оставалось отзывчивым, время от времени обрабатываем события Qt
            if (y & 31) == 0:
                QtWidgets.QApplication.processEvents()
        self.finished = True
        self.cur_scanline = self.H
        self.updated.emit()

    # получить глубину (в системе камеры) для заданного логического x_log, y_log и полигона pid
    # возвращает z_cam (меньше - ближе к камере) или None если вычислить нельзя
    def z_at(self, pid, x_log, y_log):
        plane = self.poly_planes.get(pid, None)
        if plane is None: return None
        a,b,c,d = plane

        # знаменатель выражения (выражаем из уравнения плокости)
        denom = a * x_log + b * y_log + c * self.focal
        if abs(denom) < 1e-12: return None
        s = - d / denom

        # s должен быть положительным, иначе точка за камерой или некорректна
        if s <= 0: return None
        z_cam = s * self.focal
        return float(z_cam)

    # аналитический расчёт координаты x логической при фиксированном y_log, где две плоскости пересекаются
    # используется при поиске точек разбиения интервала
    def intersection_x_of_planes(self, p1, p2, y_log):
        pl1 = self.poly_planes.get(p1, None); pl2 = self.poly_planes.get(p2, None)
        if pl1 is None or pl2 is None: return None
        a1,b1,c1,d1 = pl1; a2,b2,c2,d2 = pl2
        # система уравнений сводится к вычислению x = RHS / A
        A = d1 * a2 - d2 * a1
        RHS = (d2 * b1 - d1 * b2) * y_log + (d2 * c1 - d1 * c2) * self.focal
        if abs(A) < 1e-12: return None
        x = RHS / A
        return float(x)

    # основной алгоритм обработки одной строки (y_px - номер строки в пикселях)
    def _process_scanline(self, y_px, start_time):
        # проверка границ
        if y_px < 0 or y_px >= self.H:
            return
        
        # очищаем данные предыдущей строки
        self.current_intersections.clear(); self.current_intervals.clear()
        self.scanline_zbuffer.fill(np.inf)

        # переводим пиксельный y в логические координаты y_log (для математических расчётов)
        y_log = (self.cy - (y_px + 0.5)) / self.scale_screen

        # собираем список пересечений каждого полигона со строкой: (pid, x_log_intersection, x_px)
        intersections = []
        for p in self.scene:
            pid = p['id']
            proj = self.projected.get(pid, None)
            if proj is None:
                proj, _ = project_with_camera_and_depth(p['verts'], self.camera_pos, self.camera_target, self.camera_up, self.focal)
            n = len(proj)
            # перебираем рёбра полигона, ищем пересечение отрезка с горизонталью y_log
            for i in range(n):
                p1 = proj[i]; p2 = proj[(i+1)%n]
                if p1 is None or p2 is None: continue
                x1,y1,z1 = p1; x2,y2,z2 = p2
                # условие попадания логического y в диапазон y1..y2 (включая границы)
                if (y1 <= y_log and y_log <= y2) or (y2 <= y_log and y_log <= y1):
                    # если отрезок почти горизонтален, пропускаем (во избежание деления на ноль)
                    if abs(y2 - y1) < 1e-12:
                        continue
                    # параметрическая интерполяция для нахождения x пересечения
                    t = (y_log - y1) / (y2 - y1)
                    x_int = x1 + t * (x2 - x1)
                    # переводим в пиксельную координату и округляем для удобства логики работы с пикселями
                    x_px = int(round(self.cx + x_int * self.scale_screen))
                    intersections.append((pid, x_int, x_px))
        # сортируем пересечения по логическому x (лево -> право)
        intersections.sort(key=lambda it: it[1])
        self.current_intersections = intersections.copy()

        # если менее двух пересечений, заполнения не будет
        if len(intersections) < 2:
            return

        # проходим интервалы между соседними пересечениями, поддерживая множество активных полигонов
        active_now = set()
        for i in range(len(intersections) - 1):
            pid_i, x_i, px_i = intersections[i]
            # изменение состояния активного множества по парности (вход/выход из контура полигона)
            if pid_i in active_now:
                active_now.remove(pid_i)
            else:
                active_now.add(pid_i)

            x_left = x_i
            x_right = intersections[i+1][1]
            # если интервал пустой или вырожденный, пропускаем
            if x_right <= x_left + 1e-12:
                continue

            # переводим логические границы интервала в пиксельные координаты (целые)
            xL_px = max(0, int(math.ceil(self.cx + x_left * self.scale_screen)))
            xR_px = min(self.W - 1, int(math.floor(self.cx + x_right * self.scale_screen)))
            if xR_px < xL_px:
                xR_px = xL_px

            # если нет активных полигонов — это фон
            if len(active_now) == 0:
                self.current_intervals.append((xL_px, xR_px, None))

            # если ровно один полигон — простой случай: заполняем интервал этим полигоном
            elif len(active_now) == 1:
                pid_single = next(iter(active_now))
                self.current_intervals.append((xL_px, xR_px, pid_single))
                # для каждого пикселя интервала вычисляем глубину этого полигона и обновляем frame_z/frame_pid
                for px in range(xL_px, xR_px + 1):
                    # переводим пиксельную x в логическую координату x_log
                    x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                    z = self.z_at(pid_single, x_log_px, y_log)
                    if z is not None:
                        # обновляем локальный буфер для строки
                        if z < self.scanline_zbuffer[px]:
                            self.scanline_zbuffer[px] = z
                        # и глобальный кадровый буфер: если новое значение ближе, перезаписываем
                        if z < self.frame_z[y_px, px]:
                            self.frame_z[y_px, px] = z
                            self.frame_pid[y_px, px] = pid_single
            else:
                # если несколько полигонов перекрываются на этом интервале — используем более сложное разрешение
                self._resolve_interval(x_left, x_right, set(active_now), y_log, xL_px, xR_px, start_time, y_px)

    # рекурсивное разрешение конфликтов в интервале (вызывается для случая len(active_polys) > 1)
    def _resolve_interval(self, a_init, b_init, active_polys_set, y_log, xL_px_init, xR_px_init, start_time, y_px):
        # стек для имитации рекурсии: элементы вида (a, b, set_of_polys, depth)
        stack = [(a_init, b_init, set(active_polys_set), 0)]
        ops = 0
        splits = 0
        seen_split = set()
        while stack:
            # даём время UI обрабатывать события (важно при длительной работе)
            if (ops & 127) == 0:
                QtWidgets.QApplication.processEvents()
            ops += 1

            # защита по времени: если обработка интервала занимает слишком долго, используем центр-семплинг
            if time.time() - start_time > self.SCANLINE_TIME_LIMIT:
                # для всех оставшихся частей стека выполняем упрощённую обработку: берём лучший полигон по центру
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
                        # обновляем frame_z/frame_pid построчно для полученного полигонального выбора
                        for px in range(pxL, pxR+1):
                            x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                            z = self.z_at(best_pid, x_log_px, y_log)
                            if z is not None and z < self.frame_z[y_px, px]:
                                self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid
                return

            # извлекаем текущий подинтервал и связанные с ним полигоны
            a,b,pols,depth = stack.pop()
            pxL = max(0, int(math.ceil(self.cx + a * self.scale_screen)))
            pxR = min(self.W - 1, int(math.floor(self.cx + b * self.scale_screen)))
            if pxR < pxL: pxR = pxL
            pixel_width = max(1, pxR - pxL + 1)
            logical_width = b - a

            # базовые условия остановки разбиения:
            # если интервал маленький в пикселях, логически узкий или глубина рекурсии достигла предела
            if pixel_width <= self.MIN_PIXEL_WIDTH or logical_width < 1e-6 or depth >= self.MAX_RECURSION_DEPTH:
                xm = 0.5*(a+b)
                best_pid, best_z = None, np.inf
                for pid in pols:
                    z = self.z_at(pid, xm, y_log)
                    if z is None: continue
                    if z < best_z:
                        best_z, best_pid = z, pid
                # помечаем этот пиксельный интервал как принадлежащий best_pid
                self.current_intervals.append((pxL, pxR, best_pid))
                if best_pid is not None:
                    # обновляем глобальные буферы глубины и id
                    for px in range(pxL, pxR+1):
                        x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                        z = self.z_at(best_pid, x_log_px, y_log)
                        if z is not None and z < self.frame_z[y_px, px]:
                            self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = best_pid
                continue

            # если в интервале нет полигонов — помечаем как фон
            if len(pols) == 0:
                self.current_intervals.append((pxL, pxR, None)); continue
            # если ровно один полигон — простой случай, помечаем им весь интервал
            if len(pols) == 1:
                pid = next(iter(pols))
                self.current_intervals.append((pxL, pxR, pid))
                for px in range(pxL, pxR+1):
                    x_log_px = ((px + 0.5) - self.cx) / self.scale_screen
                    z = self.z_at(pid, x_log_px, y_log)
                    if z is not None and z < self.frame_z[y_px, px]:
                        self.frame_z[y_px, px] = z; self.frame_pid[y_px, px] = pid
                continue

            # основной путь: пытаемся найти точки разбиения интервала по пересечениям пар плоскостей
            candidate_xis = []
            pols_list = list(pols)
            for i in range(len(pols_list)):
                for j in range(i+1, len(pols_list)):
                    p1 = pols_list[i]; p2 = pols_list[j]
                    # небольшая внутренняя граница, чтобы не брать пересечения слишком близко к краям
                    eps_inner = max(1e-9, 1e-9 * logical_width)
                    xL_test = a + eps_inner; xR_test = b - eps_inner
                    # пробуем вычислить сравнения глубин слева и справа от потенциального пересечения
                    z1L = self.z_at(p1, xL_test, y_log); z2L = self.z_at(p2, xL_test, y_log)
                    z1R = self.z_at(p1, xR_test, y_log); z2R = self.z_at(p2, xR_test, y_log)
                    # если не можем вычислить глубины в тестовых точках, пропускаем пару
                    if z1L is None or z2L is None or z1R is None or z2R is None:
                        continue
                    # смотрим, поменялся ли знак разности глубин между концами интервала
                    sL = math.copysign(1.0, z1L - z2L) if abs(z1L - z2L) > 1e-12 else 0.0
                    sR = math.copysign(1.0, z1R - z2R) if abs(z1R - z2R) > 1e-12 else 0.0
                    # если порядок пересекся, это кандидат на точку раздела
                    if sL != sR:
                        xi = self.intersection_x_of_planes(p1, p2, y_log)
                        if xi is None: continue
                        tol = max(1e-9, 1e-9 * logical_width)
                        # игнорируем пересечения, лежащие слишком близко к краям интервала
                        if xi <= a + tol or xi >= b - tol:
                            continue
                        candidate_xis.append((xi, p1, p2))
            # если найдены кандидаты, разбиваем по ближайшему (левому) кандидату
            if candidate_xis:
                candidate_xis.sort(key=lambda t: t[0])
                xi, pa, pb = candidate_xis[0]
                # округляем для защиты и повторных сплитов
                key = round(xi, 9)
                if key in seen_split or splits > 20000:
                    # защитная ветка: если уже видели этот сплит или слишком много сплитов, делаем центр-семплинг
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
                # регистрируем факт сплита и добавляем две части интервала в стек для обработки
                seen_split.add(key); splits += 1
                stack.append((xi, b, set(pols), depth+1))
                stack.append((a, xi, set(pols), depth+1))
                continue

            # если кандидатов на разделение нет, используем центр-семплинг для выбора победителя
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


# Виджет Qt
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

        self.algorithm = IntervalScanline(self.scene, self.focal, self.width(), self.height(), self.scale_screen, self.camera_pos, self.camera_target, self.camera_up)
        self.algorithm.updated.connect(self.update)
        self.algorithm.rowCompleted.connect(self.on_engine_row_completed)

        self.poly_pen = QtGui.QPen(QtGui.QColor(40,40,160), 1)
        self.axis_pen_x = QtGui.QPen(QtGui.QColor(200,40,40), 2)
        self.axis_pen_y = QtGui.QPen(QtGui.QColor(40,160,40), 2)
        self.axis_pen_z = QtGui.QPen(QtGui.QColor(30,90,200), 2)

        self._recalc_camera_spherical()
        self.last_mouse = None
        self.mouse_mode = None

        # изображение-фрейм, в которое по мере прогонoв заполняются строки
        self.frame_image = None
        self._init_frame_image()
        self.show_axes = True

    # инициализация QImage для аккумулирования результатов алгоритма
    def _init_frame_image(self):
        w, h = max(1, self.width()), max(1, self.height())
        self.frame_image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        bg = QtGui.QColor(245,245,245).rgba()
        self.frame_image.fill(bg)

    # пересчитать сферические координаты камеры
    def _recalc_camera_spherical(self):
        v = self.camera_pos - self.camera_target
        r = np.linalg.norm(v)
        if r < 1e-9: r = 1e-9; v = np.array([r,0,0], dtype=float)
        self.cam_radius = r
        self.cam_el = math.asin(v[1] / r)
        self.cam_az = math.atan2(v[2], v[0])

    # аналитическая проверка видимости одного семпла ребра по глубине
    def _edge_sample_visible_by_depth(self, pid, sx, sy):
        # если алгоритм отсутствует — считаем видимым
        if self.algorithm is None:
            return True
        W = self.algorithm.W; H = self.algorithm.H
        ix = int(math.floor(sx + 0.5)); iy = int(math.floor(sy + 0.5))
        # если точка вне экрана
        if not (0 <= ix < W and 0 <= iy < H):
            # в прогрессивном режиме считаем внеэкранные семплы видимыми
            if not self.algorithm.finished:
                return True
            else:
                return False

        # если прогрессивно и строка ещё не обработана -> считаем видимым
        if (not self.algorithm.finished) and (iy >= self.algorithm.cur_scanline):
            return True

        # переводим экранные координаты в логические x_log/y_log
        cx, cy = self.width()/2.0, self.height()/2.0
        x_log = (ix + 0.5 - cx) / self.scale_screen
        y_log = (cy - (iy + 0.5)) / self.scale_screen
        z_edge = self.algorithm.z_at(pid, x_log, y_log)
        if z_edge is None:
            if not self.algorithm.finished:
                return True
            return False

        # читаем авторитетное значение глубины для пикселя
        try:
            z_frame = float(self.algorithm.frame_z[iy, ix])
        except Exception:
            z_frame = np.inf

        # меньшая глубина (z) означает ближе к камере -> видимо, если ребро ближе, чем записанный кадр
        return z_edge < z_frame - 1e-12

    # основная отрисовка виджета
    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(245,245,245))
        w,h = self.width(), self.height()
        self.scale_screen = min(w,h) * 0.5
        self.algorithm.scale_screen = self.scale_screen
        # если размер виджета изменился — обновляем параметры движка и изображение
        if (self.algorithm.W != w) or (self.algorithm.H != h):
            self.algorithm.W = w; self.algorithm.H = h
            self.algorithm.cx = w/2.0; self.algorithm.cy = h/2.0
            self.algorithm.reset()
            self._init_frame_image()

        cx, cy = w/2.0, h/2.0

        # вычисляем экранные проекции всех полигонов (для отрисовки контуров/ребер)
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

        # рисуем изображение (построчно-заполненное)
        if self.frame_image is not None:
            qp.drawImage(0, 0, self.frame_image)

        # рисуем оси, если разрешено
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

        # рисуем контуры полигонов, но семплируем каждое ребро по пикселям и выполняем аналитический тест по глубине:
        # если семпл ребра скрыт авторитетной глубиной (frame_z), то он не рисуется. Так получается эффект постепенного исчезновения ребра
        qp.setPen(self.poly_pen)
        if self.frame_image is not None:
            pid_to_rgba = {p['id']: QtGui.qRgba(p['color'].red(), p['color'].green(), p['color'].blue(), 255)
                           for p in self.scene}
            bg_rgba = QtGui.QColor(245,245,245).rgba()
            for pid, poly2d, color in final_polys_2d:
                n = len(poly2d)
                for i in range(n):
                    p1 = poly2d[i]; p2 = poly2d[(i+1)%n]
                    if p1 is None or p2 is None:
                        continue
                    sx1, sy1 = p1; sx2, sy2 = p2
                    dx = sx2 - sx1; dy = sy2 - sy1
                    seg_len = math.hypot(dx, dy)
                    if seg_len < 1e-6:
                        ix = int(round(sx1)); iy = int(round(sy1))
                        if 0 <= ix < self.frame_image.width() and 0 <= iy < self.frame_image.height():
                            # аналитический тест: проверить видимость точки ребра
                            if self._edge_sample_visible_by_depth(pid, sx1, sy1):
                                qp.drawPoint(ix, iy)
                        continue
                    # плотность семплинга примерно 1 семпл на экранный пиксель
                    samples = max(1, int(math.ceil(seg_len)))
                    for k in range(samples+1):
                        t = k / samples
                        mx = sx1 + dx * t
                        my = sy1 + dy * t
                        ix = int(math.floor(mx + 0.5))
                        iy = int(math.floor(my + 0.5))
                        if not (0 <= ix < self.frame_image.width() and 0 <= iy < self.frame_image.height()):
                            continue
                        # аналитический тест глубины: если видим, рисуем точку ребра
                        if self._edge_sample_visible_by_depth(pid, mx, my):
                            qp.drawPoint(ix, iy)
                        else:
                            # скрыто другим полигоном -> не рисуем (это обеспечивает постепенное стирание)
                            pass
        else:
            # запасной вариант: если нет frame_image, рисуем обычные линии
            for pid, poly2d, color in final_polys_2d:
                n = len(poly2d)
                for i in range(n):
                    p1 = poly2d[i]; p2 = poly2d[(i+1)%n]
                    if p1 is None or p2 is None: continue
                    qp.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))

        # индикация текущей сканирующей строки и интервалов (для прогрессивного режима)
        y_scan = self.algorithm.cur_scanline
        if 0 <= y_scan < h and self.algorithm.running:
            qp.setPen(QtGui.QPen(QtGui.QColor(200,20,20), 1, QtCore.Qt.DashLine))
            qp.drawLine(0, y_scan, w, y_scan)
            qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10), 1))
            for (pid, x_log, x_px) in self.algorithm.current_intersections:
                qp.setBrush(QtGui.QColor(255,255,255))
                qp.drawEllipse(QtCore.QPointF(x_px, y_scan), 3, 3)
                qp.drawText(x_px+4, y_scan-4, f"{pid}")
            for (xL_px, xR_px, pid) in self.algorithm.current_intervals:
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

        # статусная строка
        qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10))); qp.setFont(QtGui.QFont("Consolas", 11))
        run_state = "Выполняется" if self.algorithm.running else ("Завершён" if self.algorithm.finished else "Остановлен")
        qp.drawText(10, 20, f"Интервальный алгоритм — {run_state}    Строка: {self.algorithm.cur_scanline}/{self.height()}")

    # обработчик завершения строки
    def on_engine_row_completed(self, y):
        if self.frame_image is None:
            return
        img_w = self.frame_image.width()
        img_h = self.frame_image.height()
        if y < 0 or y >= img_h:
            return

        # подготовка преобразования id -> rgba
        pid_to_rgba = {}
        for p in self.scene:
            c = p['color']
            pid_to_rgba[p['id']] = QtGui.qRgba(c.red(), c.green(), c.blue(), 255)
        bg = QtGui.QColor(245,245,245).rgba()

        # копируем построчную информацию в изображение
        arr_pid = self.algorithm.frame_pid
        max_x = min(img_w, arr_pid.shape[1])
        for x in range(max_x):
            pid = int(arr_pid[y, x])
            if pid >= 0:
                self.frame_image.setPixel(x, y, pid_to_rgba.get(pid, bg))
            else:
                self.frame_image.setPixel(x, y, bg)

        # дополнительно рисуем интервалы, вычисленные алгоритмом для этой строки
        for (xL_px, xR_px, pid) in self.algorithm.current_intervals:
            if pid is None: continue
            xL = max(0, xL_px); xR = min(img_w-1, xR_px)
            if xL > xR: continue
            color_rgba = pid_to_rgba.get(pid, bg)
            for x in range(xL, xR+1):
                self.frame_image.setPixel(x, y, color_rgba)

        # обновляем только узкую полоску в виджете 
        self.update(0, y, max_x, 2)

    # начало поворота камеры (левый клик)
    def mousePressEvent(self, ev):
        if self.algorithm.running:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self.show_axes = True
            if self.algorithm.finished:
                self.algorithm.finished = False
                self.algorithm.frame_pid.fill(-1)
                self.algorithm.frame_z.fill(np.inf)
                self._init_frame_image()
            self.mouse_mode = "orbit_cam"
            v = self.camera_pos - self.camera_target
            r = np.linalg.norm(v)
            if r < 1e-9: r = 1e-9; v = np.array([r,0,0])
            self._orbit_start = (r, math.asin(v[1]/r), math.atan2(v[2], v[0]))
            self._orbit_start_mouse = ev.pos()
        self.last_mouse = ev.pos()

    # обработка перемещения мыши при вращении камеры
    def mouseMoveEvent(self, ev):
        # во время работы алгоритма запрещаем вращать камеру
        if self.algorithm.running:
            return
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

            # при повороте камеры сбрасываем накопленное изображение и пересчитываем проекции
            self.algorithm.reset()
            self._init_frame_image()
            self.show_axes = True
            self.algorithm.set_camera(self.camera_pos)
            self._recalc_camera_spherical()
            self.update()
        self.last_mouse = pos

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None
        self.last_mouse = None

    # клавиши: R - сброс алгоритма, Space - мгновенная работа алгоритма, S - шаг построчно
    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            self.reset_algorithm()
        elif k == QtCore.Qt.Key_Space:
            if self.algorithm.finished:
                self.reset_algorithm()
            self.start_instant()
        elif k == QtCore.Qt.Key_S:
            if not self.algorithm.running:
                if self.algorithm.finished:
                    self.reset_algorithm()
                self.algorithm.on_tick()
        self.update()

    # сброс алгоритма: очищаем кадр, возвращаем строку в 0, показываем оси
    def reset_algorithm(self):
        self.algorithm.reset()
        self._init_frame_image()
        self.show_axes = True
        self.update()

    # старт прогрессивного режима (анимация построчного закрашивания)
    def start_progressive(self):
        if self.algorithm.finished:
            self.reset_algorithm()
        self.show_axes = False
        self.algorithm.start()
        self.update()

    # остановка прогрессивного режима
    def stop_progressive(self):
        self.algorithm.pause()
        self.update()

    # мгновенная полная отрисовка
    def start_instant(self):
        if self.algorithm.finished:
            self.reset_algorithm()
        self.show_axes = False
        self.algorithm.run_full()
        self.update()

# Главное окно приложения
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 2 - Интервальный алгоритм построчного сканирования")
        self.gl = GLWidget()
        self.setCentralWidget(self.gl)
        self.create_controls()
        self.resize(1300, 780)

    # создание панели управления
    def create_controls(self):
        panel = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel("<b>Управление алгоритмом</b>"))

        btn_instant = QtWidgets.QPushButton("Выполнить мгновенно (Space)")
        btn_instant.clicked.connect(lambda: self.on_instant_clicked())
        btn_progressive = QtWidgets.QPushButton("Запустить прогрессивно")
        btn_progressive.clicked.connect(lambda: self.on_progressive_clicked())
        btn_stop = QtWidgets.QPushButton("Остановить прогрессивно")
        btn_stop.clicked.connect(lambda: self.on_stop_clicked())
        btn_step = QtWidgets.QPushButton("Шаг — 1 строка (S)")
        btn_step.clicked.connect(lambda: self.on_step_clicked())
        btn_reset_alg = QtWidgets.QPushButton("Сброс алгоритма (R)")
        btn_reset_alg.clicked.connect(lambda: self.on_reset_clicked())

        layout.addWidget(btn_instant); layout.addWidget(btn_progressive); layout.addWidget(btn_stop)
        layout.addWidget(btn_step); layout.addWidget(btn_reset_alg)

        layout.addWidget(QtWidgets.QLabel("<b>Сцены</b>"))
        btn_demo = QtWidgets.QPushButton("Сгенерировать демо-сцену (прямоугольник + треугольник)")
        btn_demo.clicked.connect(self.on_demo_scene)
        btn_random = QtWidgets.QPushButton("Сгенерировать случайную сцену (5-6 многоугольников)")
        btn_random.clicked.connect(self.on_random_scene)
        layout.addWidget(btn_demo); layout.addWidget(btn_random)

        instr = QtWidgets.QLabel(
            "- ЛКМ + перетаскивание: вращение камеры\n"
            "- Выполнить мгновенно: полная отрисовка\n"
            "- Запустить прогрессивно: анимированная отрисовка построчно\n"
            "- Остановить прогрессивно: приостановка\n"
            "- S: выполнить 1 строку вручную\n"
            "- R: сброс алгоритма (очистка закрашенных фрагментов, возврат строки в 0)\n"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)
        layout.addStretch()
        panel.setLayout(layout)

        dock = QtWidgets.QDockWidget("Панель управления", self)
        dock.setWidget(panel); self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # горячие клавиши
        self.shortcut_space = QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self)
        self.shortcut_space.activated.connect(lambda: self.on_instant_clicked())
        self.shortcut_s = QtWidgets.QShortcut(QtGui.QKeySequence("S"), self)
        self.shortcut_s.activated.connect(lambda: self.on_step_clicked())
        self.shortcut_r = QtWidgets.QShortcut(QtGui.QKeySequence("R"), self)
        self.shortcut_r.activated.connect(lambda: self.on_reset_clicked())

    # обработчики кнопок панели
    def on_instant_clicked(self):
        if self.gl.algorithm.finished:
            self.gl.reset_algorithm()
        self.gl.start_instant()

    def on_progressive_clicked(self):
        if self.gl.algorithm.finished:
            self.gl.reset_algorithm()
        self.gl.start_progressive()

    def on_stop_clicked(self):
        self.gl.stop_progressive()

    def on_step_clicked(self):
        if self.gl.algorithm.running:
            return
        if self.gl.algorithm.finished:
            self.gl.reset_algorithm()
        self.gl.algorithm.on_tick()
        self.gl.update()

    def on_reset_clicked(self):
        self.gl.reset_algorithm()

    # генерация демо-сцены
    def on_demo_scene(self):
        self.gl.scene = generate_demo_scene()
        self.gl.algorithm.scene = self.gl.scene
        self.gl.algorithm._prepare_projection()
        self.gl.reset_algorithm()
        self.gl.update()

    # генерация случайной сцены
    def on_random_scene(self):
        self.gl.scene = generate_random_scene(random.randint(5,6))
        self.gl.algorithm.scene = self.gl.scene
        self.gl.algorithm._prepare_projection()
        self.gl.reset_algorithm()
        self.gl.update()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow(); w.show()
sys.exit(app.exec_())