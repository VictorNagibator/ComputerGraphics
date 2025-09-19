import sys
import math
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

MIN_SCALE = 0.2  # минимальный масштаб модели
MAX_SCALE = 4.0  # максимальный масштаб модели

# Матрицы преобразований
def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    # Матрица переноса в однородных координатах
    M = np.eye(4, dtype=float) # создаем матрицу 4x4 с единицами на главной диагонали
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M

def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    # Матрица масштабирования по осям
    M = np.eye(4, dtype=float)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M

def rotation_x(theta: float) -> np.ndarray:
    # Матрица поворота вокруг OX.
    # Формула:
    #  [1   0      0    0]
    #  [0  cos  -sin    0]
    #  [0  sin   cos    0]
    #  [0   0     0     1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[1,1], M[1,2] = c, -s
    M[2,1], M[2,2] = s, c
    return M

def rotation_y(theta: float) -> np.ndarray:
    # Матрица поворота вокруг OY.
    # Формула:
    #  [ cos   0   sin  0]
    #  [  0    1    0   0]
    #  [-sin   0   cos  0]
    #  [  0    0    0   1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0,0], M[0,2] = c, s
    M[2,0], M[2,2] = -s, c
    return M

def rotation_z(theta: float) -> np.ndarray:
    # Матрица поворота вокруг OZ.
    # Формула:
    #  [cos -sin  0  0]
    #  [sin  cos  0  0]
    #  [ 0    0   1  0]
    #  [ 0    0   0  1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0,0], M[0,1] = c, -s
    M[1,0], M[1,1] = s, c
    return M

def look_at_matrix(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    # Строим матрицу вида (мировые -> камера)
    # camera_pos: положение камеры в мировых координатах
    # target: точка, на которую смотрит камера
    # up: вектор "вверх" (обычно (0,1,0))
    #
    # Алгоритм:
    # 1) zc = normalize(target - camera) — направление взгляда камеры
    # 2) xc = normalize(cross(zc, up)) — правая ось камеры
    # 3) yc = cross(xc, zc) — вверх камеры (ортогонален)
    # 4) Формируем матрицу из базиса (xc, yc, zc) и переводим так, чтобы камера встала в начале координат
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
    V[0,0:3] = xc
    V[1,0:3] = yc
    V[2,0:3] = zc
    # смещаем так, чтобы камера оказалась в начале системы координат камеры
    V[0:3,3] = -V[0:3,0:3] @ C
    return V

def project_with_camera(points4, camera_pos, target, up, focal):
    # Проецируем вершины из мировых в экранные 2D координаты через камеру
    # Шаги:
    # 1) мировые -> камера: умножаем на look_at_matrix
    # 2) В координатах камеры используем простую перспективную проекцию на плоскость z = focal_cam:
    #    для точки в (x_cam, y_cam, z_cam):
    #      если z_cam <= 0: точка за камерой или на ней -> пропускаем (None)
    #      коэффициент t = focal / z_cam
    #      экранные логические x' = t * x_cam, y' = t * y_cam
    #
    # Параметр focal — фокус камеры: чем больше — тем менее выражена перспектива
    pts = np.array(points4, dtype=float)
    V = look_at_matrix(np.array(camera_pos, dtype=float),
                       np.array(target, dtype=float),
                       np.array(up, dtype=float))
    pts_cam = (V @ pts.T).T
    res = []
    eps = 1e-9
    for p in pts_cam:
        x, y, z, w = p
        # если точка позади камеры или слишком близко — не рисуем
        if z <= eps:
            res.append(None)
            continue
        t = focal / z
        res.append((x * t, y * t))
    return np.array(res, dtype=object)

def make_letter_V(size=1.0, depth=0.2):
    # Создаём проволочную букву V как набор вершин и рёбер
    # Вершины задаются в локальной системе модели в однородных координатах (x,y,z,1)
    h = size
    w = size * 0.5
    top_left = (-w, h/2)
    top_right = (w, h/2)
    bottom = (0, -h/2)
    z0, z1 = depth/2, -depth/2  # передняя / задняя
    verts = [
        (top_left[0], top_left[1], z0, 1.0),   # 0 передняя левая
        (top_right[0], top_right[1], z0, 1.0), # 1 передняя правая
        (bottom[0], bottom[1], z0, 1.0),       # 2 передняя нижняя
        (top_left[0], top_left[1], z1, 1.0),   # 3 задняя левая
        (top_right[0], top_right[1], z1, 1.0), # 4 задняя правая
        (bottom[0], bottom[1], z1, 1.0),       # 5 задняя нижняя
    ]
    # Рёбра: пары индексов вершин, которые соединяем линией.
    edges = [
        (0,2), (1,2),       # передняя плоскость
        (3,5), (4,5),       # задняя плоскость
        (0,3), (1,4), (2,5) # соединения перед/зад
    ]
    return np.array(verts, dtype=float), edges

class GLWidget(QtWidgets.QWidget):
    # Сигнал об изменении камеры
    cameraChanged = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # буква V
        self.model_vertices, self.model_edges = make_letter_V(size=1.0, depth=0.25)

        # модель в мировых координатах: перенос/поворот/масштаб
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0  # перенос
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0  # повороты
        self.s = 1.0 # общий масштаб

        # ввод мышью
        self.last_mouse = None
        self.mouse_mode = None

        # камера: позиция, цель и вектор "вверх"
        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)  # где находится камера в мире
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)  # куда смотрит
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=float)  # вверх камеры
        self.focal = 4.0 # фокус для проекции

        # для вращения камеры вокруг объекта храним сферические параметры
        self._recalc_camera_spherical()

        # параметры спиральной анимации
        self.anim_running = False
        self.anim_start_time = None
        self.omega0 = 6.0 # начальная угловая скорость (рад/с)
        self.radius = 0.6 # целевой радиус спирали 
        self.vz = -1.2  # смещение вдоль выбранной оси за всю анимацию
        self.anim_axis = 'OZ' # выбранная ось: 'OX','OY','OZ'
        self.anim_duration = 6.0  # длительность анимации (сек)

        # таймер анимации
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self.on_anim_tick)
        self.anim_timer.start(30)

        # текущие вычисления для отображения
        self.current_angular_velocity = 0.0
        self.current_radius = 0.0

        # ручки для рисования (разные цвета)
        self.line_pen = QtGui.QPen(QtGui.QColor(40, 40, 180), 2)
        self.axis_pen_x = QtGui.QPen(QtGui.QColor(200, 30, 30), 2) # красная X
        self.axis_pen_y = QtGui.QPen(QtGui.QColor(30, 160, 30), 2) # зелёная Y
        self.axis_pen_z = QtGui.QPen(QtGui.QColor(30, 120, 200), 2) # синяя Z

        # позиция модели в момент старта анимации (чтобы двигаться относительно неё)
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float)

    def _recalc_camera_spherical(self):
        # Пересчитываем сферические параметры (работаем с сферической системе координат)

        # по текущей camera_pos и camera_target
        v = self.camera_pos - self.camera_target
        r = np.linalg.norm(v) # радиус (расстояние до камеры)
        if r < 1e-6:
            r = 1e-6
            v = np.array([r, 0.0, 0.0])
        el = math.asin(v[1] / r)  # угол подъёма
        az = math.atan2(v[2], v[0]) # горизонтальный угол
        self.cam_radius = r
        self.cam_azimuth = az
        self.cam_elevation = el

    def _update_camera_from_spherical(self):
        # Пересчитываем camera_pos по сферическим параметрам
        r = max(1e-6, self.cam_radius)
        el = self.cam_elevation
        az = self.cam_azimuth
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        # получаем абсолютную позицию в мировых координатах
        self.camera_pos = np.array([x, y, z], dtype=float) + self.camera_target
        # уведомляем интерфейс что камера изменилась
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))

    def sizeHint(self):
        return QtCore.QSize(900, 700)

    def model_matrix(self):
        # Собираем матрицу модели M = T * Rz * Ry * Rx * S
        # Порядок: масштаб -> повороты -> перенос (в мировых координатах)
        T = translation_matrix(self.tx, self.ty, self.tz)
        Rx, Ry, Rz = rotation_x(self.rx), rotation_y(self.ry), rotation_z(self.rz)
        S = scale_matrix(self.s, self.s, self.s)
        return T @ (Rz @ (Ry @ (Rx @ S)))

    def paintEvent(self, event):
        # Рисуем сцену: модель, мировые оси, интерфейс
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(245,245,245))

        w, h = self.width(), self.height()
        cx, cy = w/2.0, h/2.0

        # применяем матрицу модели к вершинам (получаем мировые координаты вершин)
        M = self.model_matrix()
        verts4_world = (M @ self.model_vertices.T).T

        # проецируем через камеру
        pts2d_model = project_with_camera(verts4_world, self.camera_pos, self.camera_target, self.camera_up, self.focal)

        # переводим логические x,y в пиксели
        scale_screen = min(w, h) * 0.5

        # рисуем рёбра модели
        qp.setPen(self.line_pen)
        for (i, j) in self.model_edges:
            p1 = pts2d_model[i]; p2 = pts2d_model[j]
            if p1 is None or p2 is None:
                continue
            sx1, sy1 = cx + p1[0]*scale_screen, cy - p1[1]*scale_screen
            sx2, sy2 = cx + p2[0]*scale_screen, cy - p2[1]*scale_screen
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

        # рисуем мировые оси (чтобы видеть систему координат мира)
        axis_len = 1.2
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        axis_points = [
            (origin, np.array([axis_len, 0.0, 0.0, 1.0])),  # X
            (origin, np.array([0.0, axis_len, 0.0, 1.0])),  # Y
            (origin, np.array([0.0, 0.0, axis_len, 1.0])),  # Z
        ]
        pts_for_axes = np.array([p for pair in axis_points for p in pair], dtype=float)
        proj_axes = project_with_camera(pts_for_axes, self.camera_pos, self.camera_target, self.camera_up, self.focal)
        for idx, pen in enumerate((self.axis_pen_x, self.axis_pen_y, self.axis_pen_z)):
            p_origin = proj_axes[2*idx]
            p_end = proj_axes[2*idx + 1]
            if p_origin is None or p_end is None:
                continue
            sx1, sy1 = cx + p_origin[0]*scale_screen, cy - p_origin[1]*scale_screen
            sx2, sy2 = cx + p_end[0]*scale_screen, cy - p_end[1]*scale_screen
            qp.setPen(pen)
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
            # стрелочки на концах осей для наглядности
            dx = sx2 - sx1; dy = sy2 - sy1
            L = math.hypot(dx, dy)
            if L > 1.0:
                ux, uy = dx / L, dy / L
                vx, vy = -uy, ux
                arrow_len = min(12, L*0.15)
                ax1 = sx2 - ux*arrow_len + vx*arrow_len*0.4
                ay1 = sy2 - uy*arrow_len + vy*arrow_len*0.4
                ax2 = sx2 - ux*arrow_len - vx*arrow_len*0.4
                ay2 = sy2 - uy*arrow_len - vy*arrow_len*0.4
                qp.drawLine(int(sx2), int(sy2), int(ax1), int(ay1))
                qp.drawLine(int(sx2), int(sy2), int(ax2), int(ay2))

        # выводим текущие параметры модели и камеры
        qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10)))
        qp.setFont(QtGui.QFont("Consolas", 10))
        txt = (f"Позиция=({self.tx:.2f}, {self.ty:.2f}, {self.tz:.2f})    "
               f"Поворот(градусы)=({math.degrees(self.rx):.1f}, {math.degrees(self.ry):.1f}, {math.degrees(self.rz):.1f})    "
               f"Масштаб={self.s:.2f}")
        qp.drawText(10, 20, txt)
        camtxt = f"Камера=({self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f})"
        qp.drawText(10, 40, camtxt)
        anim_state = "Выполняется" if self.anim_running else "Остановлена"
        qp.drawText(10, 60, f"Анимация: {anim_state}    Ось: {self.anim_axis}")
        qp.drawText(10, 80, f"Угловая скорость={self.current_angular_velocity:.3f} рад/с    Радиус={self.current_radius:.3f}")

    def mousePressEvent(self, ev):
        # ЛКМ + Ctrl -> вращение камеры; ЛКМ без Ctrl -> вращение модели; ПКМ -> перемещение модели
        if ev.button() == QtCore.Qt.LeftButton:
            mods = QtWidgets.QApplication.keyboardModifiers()
            if mods & QtCore.Qt.ControlModifier:
                # вращение камеры: запоминаем стартовые значения
                self.mouse_mode = "orbit_cam"
                self._orbit_start_az = self.cam_azimuth
                self._orbit_start_el = self.cam_elevation
                self._orbit_start_mouse = ev.pos()
            else:
                # вращение модели
                self.mouse_mode = "rotate"
            self.last_mouse = ev.pos()
        elif ev.button() == QtCore.Qt.RightButton:
            # перемещение модели
            self.mouse_mode = "pan"
            self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        # Обработка движения мыши в зависимости от режима
        if self.last_mouse is None:
            return
        pos = ev.pos()
        dx = pos.x() - self.last_mouse.x()
        dy = pos.y() - self.last_mouse.y()
        mods = QtWidgets.QApplication.keyboardModifiers()

        if self.mouse_mode == "rotate":
            # Вращение модели: ЛКМ двигает по OY (dx) и OX (dy)
            if mods & QtCore.Qt.ShiftModifier:
                # Shift+ЛКМ — вращение вокруг OZ
                self.rz += dx * 0.01
            else:
                self.ry += dx * 0.01
                self.rx += dy * 0.01

        elif self.mouse_mode == "pan":
            # Перемещение модели: ПКМ двигает по XY, Shift+ПКМ — по Z
            if mods & QtCore.Qt.ShiftModifier:
                self.tz += dy * 0.005
            else:
                self.tx += dx * 0.005
                self.ty -= dy * 0.005

        elif self.mouse_mode == "orbit_cam":
            # Вращение камеры вокруг объекта
            start = self._orbit_start_mouse
            ddx = pos.x() - start.x()
            ddy = pos.y() - start.y()
            az = self._orbit_start_az + ddx * 0.01
            el = self._orbit_start_el + ddy * 0.01

            # ограничиваем, чтобы камера не переворачивалась
            max_el = math.radians(89.0)
            el = max(-max_el, min(max_el, el))
            self.cam_azimuth = az
            self.cam_elevation = el
            # пересчитываем позицию камеры по сферическим координатам
            self._update_camera_from_spherical()

        self.last_mouse = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        # Сбрасываем режим мыши
        self.mouse_mode = None
        self.last_mouse = None

    def wheelEvent(self, ev):
        # Колесо мыши масштабирует модель
        delta = ev.angleDelta().y() / 120.0
        factor = 1.1 ** delta
        self.s *= factor
        self.s = max(MIN_SCALE, min(MAX_SCALE, self.s))
        self.update()

    def keyPressEvent(self, ev):
        # R — сброс, Space — старт/пауза анимации
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            self.reset_transform()
        elif k == QtCore.Qt.Key_Space:
            self.toggle_animation()
        self.update()

    def start_animation(self):
        # Запоминаем позицию модели в момент старта, включаем таймер
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float)
        self.anim_running = True
        self.anim_start_time = time.time()
        self.current_angular_velocity = self.omega0
        self.current_radius = 0.0
        self.update()

    def stop_animation(self):
        # Останов анимации
        self.anim_running = False
        self.anim_start_time = None
        self.update()

    def toggle_animation(self):
        # Переключаем состояние анимации
        if self.anim_running:
            self.stop_animation()
        else:
            self.start_animation()

    def on_anim_tick(self):
        # Спиральная анимация с линейным замедлением
        if not self.anim_running:
            return
        t = time.time() - self.anim_start_time
        T = max(self.anim_duration, 1e-6)
        if t >= T:
            progress = 1.0
            finished = True
        else:
            progress = t / T
            finished = False

        # угловая скорость линейно падает до 0
        omega_t = self.omega0 * max(0.0, 1.0 - progress)

        # накопленный угол (интеграл omega(t)): theta = omega0*(t - t^2/(2T))
        if not finished:
            theta = self.omega0 * (t - (t*t) / (2.0 * T))
        else:
            theta = 0.5 * self.omega0 * T

        # радиус растёт линейно от 0 до целевого радиуса
        radius_t = self.radius * progress

        base = np.array(self._base_pos, dtype=float)

        axis = self.anim_axis.upper()
        if axis == 'OZ':
            # спираль в плоскости XY, смещение по мировой Z
            x = base[0] + radius_t * math.cos(theta)
            y = base[1] + radius_t * math.sin(theta)
            z = base[2] + self.vz * progress
            # поворот модели: вращаем вокруг OZ (локальный угол rZ)
            self.rz = theta
        elif axis == 'OY':
            # спираль в XZ, смещение по мировой Y
            x = base[0] + radius_t * math.cos(theta)
            z = base[2] + radius_t * math.sin(theta)
            y = base[1] + self.vz * progress
            self.ry = theta
        else:  # 'OX'
            # спираль в YZ, смещение по мировой X
            y = base[1] + radius_t * math.cos(theta)
            z = base[2] + radius_t * math.sin(theta)
            x = base[0] + self.vz * progress
            self.rx = theta

        # Записываем новые мировые координаты модели
        self.tx, self.ty, self.tz = float(x), float(y), float(z)

        # Обновляем интерфейс
        self.current_angular_velocity = omega_t
        self.current_radius = radius_t

        if finished:
            self.stop_animation()

        self.update()

    def reset_transform(self):
        # Сбрасываем модельные трансформации в ноль
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0
        self.current_angular_velocity = 0.0
        self.current_radius = 0.0
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float)

        # Сбрасываем камеру на стандартную позицию (4,4,4)
        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self._recalc_camera_spherical()

        # обновляем интерфейс
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))
        self.update()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1")
        self.gl = GLWidget(self)
        self.setCentralWidget(self.gl)
        self.create_controls()

    def create_controls(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(QtWidgets.QLabel("<b>Параметры анимации</b>"))

        layout.addWidget(QtWidgets.QLabel("Ось вращения при анимации:"))
        axis_box = QtWidgets.QComboBox()
        axis_box.addItems(["OX", "OY", "OZ"])
        axis_box.setCurrentText(self.gl.anim_axis)
        axis_box.currentTextChanged.connect(lambda t: setattr(self.gl, "anim_axis", t))
        layout.addWidget(axis_box)

        layout.addWidget(QtWidgets.QLabel("Начальная угловая скорость (рад/с):"))
        omega_spin = QtWidgets.QDoubleSpinBox()
        omega_spin.setRange(0.1, 40.0)
        omega_spin.setValue(self.gl.omega0)
        omega_spin.setSingleStep(0.1)
        omega_spin.valueChanged.connect(lambda v: setattr(self.gl, "omega0", v))
        layout.addWidget(omega_spin)

        layout.addWidget(QtWidgets.QLabel("Целевой радиус спирали:"))
        radius_spin = QtWidgets.QDoubleSpinBox()
        radius_spin.setRange(0.0, 3.0)
        radius_spin.setValue(self.gl.radius)
        radius_spin.setSingleStep(0.05)
        radius_spin.valueChanged.connect(lambda v: setattr(self.gl, "radius", v))
        layout.addWidget(radius_spin)

        layout.addWidget(QtWidgets.QLabel("Итоговое смещение:"))
        vz_spin = QtWidgets.QDoubleSpinBox()
        vz_spin.setRange(-5.0, 5.0)
        vz_spin.setValue(self.gl.vz)
        vz_spin.setSingleStep(0.05)
        vz_spin.valueChanged.connect(lambda v: setattr(self.gl, "vz", v))
        layout.addWidget(vz_spin)

        layout.addWidget(QtWidgets.QLabel("Длительность анимации (сек):"))
        dur_spin = QtWidgets.QDoubleSpinBox()
        dur_spin.setRange(0.5, 60.0)
        dur_spin.setValue(self.gl.anim_duration)
        dur_spin.setSingleStep(0.5)
        dur_spin.valueChanged.connect(lambda v: setattr(self.gl, "anim_duration", v))
        layout.addWidget(dur_spin)

        layout.addWidget(QtWidgets.QLabel("<b>Параметры камеры</b>"))
        cam_layout = QtWidgets.QGridLayout()
        cam_layout.addWidget(QtWidgets.QLabel("Cam X:"), 0, 0)
        self.cam_x = QtWidgets.QDoubleSpinBox(); self.cam_x.setRange(-100.0, 100.0)
        self.cam_x.setValue(self.gl.camera_pos[0])
        cam_layout.addWidget(self.cam_x, 0, 1)
        cam_layout.addWidget(QtWidgets.QLabel("Cam Y:"), 1, 0)
        self.cam_y = QtWidgets.QDoubleSpinBox(); self.cam_y.setRange(-100.0, 100.0)
        self.cam_y.setValue(self.gl.camera_pos[1])
        cam_layout.addWidget(self.cam_y, 1, 1)
        cam_layout.addWidget(QtWidgets.QLabel("Cam Z:"), 2, 0)
        self.cam_z = QtWidgets.QDoubleSpinBox(); self.cam_z.setRange(-100.0, 100.0)
        self.cam_z.setValue(self.gl.camera_pos[2])
        cam_layout.addWidget(self.cam_z, 2, 1)

        def cam_changed(_):
            # При изменении обновляем позицию камеры и пересчитываем сферические параметры
            self.gl.camera_pos = np.array([self.cam_x.value(), self.cam_y.value(), self.cam_z.value()], dtype=float)
            self.gl._recalc_camera_spherical()
            self.gl.cameraChanged.emit(float(self.gl.camera_pos[0]), float(self.gl.camera_pos[1]), float(self.gl.camera_pos[2]))
            self.gl.update()
        self.cam_x.valueChanged.connect(cam_changed)
        self.cam_y.valueChanged.connect(cam_changed)
        self.cam_z.valueChanged.connect(cam_changed)

        # Синхронизируем при изменении камеры из самого виджета (орбита)
        def on_camera_changed(x, y, z):
            self.cam_x.blockSignals(True); self.cam_y.blockSignals(True); self.cam_z.blockSignals(True)
            self.cam_x.setValue(x); self.cam_y.setValue(y); self.cam_z.setValue(z)
            self.cam_x.blockSignals(False); self.cam_y.blockSignals(False); self.cam_z.blockSignals(False)
        self.gl.cameraChanged.connect(on_camera_changed)

        layout.addLayout(cam_layout)

        btn_row = QtWidgets.QHBoxLayout()
        btn_start = QtWidgets.QPushButton("Старт/Пауза (Space)")
        btn_start.clicked.connect(self.gl.toggle_animation)
        btn_reset = QtWidgets.QPushButton("Сброс (R)")
        btn_reset.clicked.connect(self.gl.reset_transform)
        btn_row.addWidget(btn_start); btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        instr = QtWidgets.QLabel(
            "Управление:\n"
            "- ЛКМ + перетаскивание: вращение модели (OX/OY)\n"
            "- Shift + ЛКМ: вращение модели вокруг OZ\n"
            "- ПКМ + перетаскивание: перенос модели по XY\n"
            "- Shift + ПКМ: перенос модели вдоль OZ\n"
            "- Колесо мыши: масштаб модели\n"
            "- Ctrl + ЛКМ: вращение камеры вокруг модели\n"
            "- Space: запустить/приостановить анимацию\n"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        layout.addStretch()
        dock = QtWidgets.QDockWidget("Управление", self)
        panel.setLayout(layout)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.resize(1200, 760)
w.show()
sys.exit(app.exec_())