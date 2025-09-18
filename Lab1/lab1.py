import sys
import math
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


MIN_SCALE = 0.2 # минимальный масштаб
MAX_SCALE = 4.0 # максимальный масштаб

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

# Перспективная проекция
def perspective_project(points4, c=4.0):
    # Центральная (перспективная) проекция:
    # - камера (центр проецирования) в точке C = (0,0,c)
    # - плоскость проекции — z = 0
    # Тогда для точки (x,y,z):
    #    t = c / (c - z)
    #    x' = t * x,  y' = t * y
    # если (c - z) близко к нулю (точка на линии взгляда камеры или за камерой),
    # точку мы обозначаем как недоступную

    pts = np.array(points4, dtype=float)
    projected = []
    eps = 1e-6
    for p in pts:
        x, y, z, w = p

        denom = c - z 
        if abs(denom) < eps:
            # точка находится на плоскости, проходящей через центр проекции — не проецируем нормально
            projected.append(None)
            continue

        koef = c / denom
        projected.append((x * koef, y * koef))
    return np.array(projected, dtype=object)


# проволочная буква V
def make_letter_V(size=1.0, depth=0.2):
    h = size
    w = size * 0.5
    top_left = (-w, h/2)
    top_right = (w, h/2)
    bottom = (0, -h/2)

    z0, z1 = depth/2, -depth/2 # передняя и задняя части
    verts = [
        (top_left[0], top_left[1], z0, 1.0),   # 0 передняя левая
        (top_right[0], top_right[1], z0, 1.0), # 1 передняя правая
        (bottom[0], bottom[1], z0, 1.0),       # 2 передняя нижняя
        (top_left[0], top_left[1], z1, 1.0),   # 3 задняя левая
        (top_right[0], top_right[1], z1, 1.0), # 4 задняя правая
        (bottom[0], bottom[1], z1, 1.0),       # 5 задняя нижняя
    ]
    edges = [
        (0,2), (1,2), # передняя плоскость V
        (3,5), (4,5), # задняя плоскость V
        (0,3), (1,4), (2,5) # соединения перед/зад (глубина)
    ]
    return np.array(verts, dtype=float), edges

# Виджет отрисовки и управления
class GLWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # буква V
        self.model_vertices, self.model_edges = make_letter_V(size=1.0, depth=0.25)

        # изначальные координаты модели
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0

        # ввод мыши
        self.last_mouse = None
        self.mouse_mode = None

        # проекция: камера в (0,0,c)
        # положительное значение фокуса -> камера находится перед плоскостью z=0 против направлению OZ
        self.focal = 4.0

        # параметры анимации
        self.anim_running = False
        self.anim_start_time = None
        self.omega0 = 6.0 # начальная угловая скорость (рад/с)
        self.radius = 0.6 # целевой радиус спирали
        self.vz = -1.2 # итоговое смещение вдоль выбранной оси
        self.anim_axis = 'OZ' # 'OX','OY','OZ'
        self.anim_duration = 6.0 # длительность анимации

        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self.on_anim_tick)
        self.anim_timer.start(30)

        # текущие значения угловой скорости и радиуса (обновляются в on_anim_tick)
        self.current_angular_velocity = 0.0
        self.current_radius = 0.0

        # отрисовка линий
        self.line_pen = QtGui.QPen(QtGui.QColor(40, 40, 180), 2)

        # храним начальную позицию при старте анимации (чтобы анимация была относительной, а не с начала координат)
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float)

    def sizeHint(self):
        return QtCore.QSize(900, 700)

    def model_matrix(self):
        # Комбинация матриц: сначала масштаб (S), затем повороты Rx, Ry, Rz, затем перенос T
        # Итог: M = T * Rz * Ry * Rx * S
        T = translation_matrix(self.tx, self.ty, self.tz)
        Rx, Ry, Rz = rotation_x(self.rx), rotation_y(self.ry), rotation_z(self.rz)
        S = scale_matrix(self.s, self.s, self.s)
        return T @ (Rz @ (Ry @ (Rx @ S))) # умножение матриц в numpy :)

    def paintEvent(self, event):
        # Отрисовка сцены: модель и интерфейс
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(245,245,245))

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # применяем матрицу модели к вершинам
        M = self.model_matrix()
        verts4 = (M @ self.model_vertices.T).T
        pts2d = perspective_project(verts4, c=self.focal)

        # масштаб для преобразования логических координат в экранные пиксели
        scale_screen = min(w, h) * 0.5

        # экранные точки
        screen_pts = [(cx + x*scale_screen, cy - y*scale_screen) for (x,y) in pts2d]

        # рисуем рёбра модели
        qp.setPen(self.line_pen)
        for (i, j) in self.model_edges:
            p1 = screen_pts[i]; p2 = screen_pts[j]
            qp.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # интерфейс: позиция / угол / масштаб / состояние анимации / текущая скорость и радиус
        qp.setPen(QtGui.QPen(QtGui.QColor(10,10,10)))
        qp.setFont(QtGui.QFont("Consolas", 10))
        txt = (f"Позиция=({self.tx:.2f}, {self.ty:.2f}, {self.tz:.2f})    "
               f"Поворот(градусы)=({math.degrees(self.rx):.1f}, {math.degrees(self.ry):.1f}, {math.degrees(self.rz):.1f})    "
               f"Масштаб={self.s:.2f}")
        qp.drawText(10, 20, txt)
        anim_state = "Выполняется" if self.anim_running else "Остановлена"
        qp.drawText(10, 40, f"Анимация: {anim_state}    Ось: {self.anim_axis}")
        # текущие значения
        qp.drawText(10, 60, f"Угловая скорость={self.current_angular_velocity:.3f} рад/с    Радиус={self.current_radius:.3f}")

    # Ввод мышью
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouse_mode = "rotate" # поворот
            self.last_mouse = ev.pos()
        elif ev.button() == QtCore.Qt.RightButton:
            self.mouse_mode = "pan" # перемещение
            self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_mouse is None:
            return
        
        pos = ev.pos()
        dx = pos.x() - self.last_mouse.x() 
        dy = pos.y() - self.last_mouse.y()
        mods = QtWidgets.QApplication.keyboardModifiers()

        if self.mouse_mode == "rotate":
            if mods & QtCore.Qt.ShiftModifier:
                # Shift+ЛКМ: вращение вокруг OZ
                # (0.01 - чтобы резко не вращалось)
                self.rz += dx * 0.01
            else:
                # ЛКМ: поворот вокруг OX и OY
                self.ry += dx * 0.01
                self.rx += dy * 0.01
        elif self.mouse_mode == "pan":
            if mods & QtCore.Qt.ShiftModifier:
                # Shift+ПКМ: движение по Z
                self.tz += dy * 0.005
            else:
                # ПКМ: движение в XY
                self.tx += dx * 0.005
                self.ty -= dy * 0.005

        self.last_mouse = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None
        self.last_mouse = None

    def wheelEvent(self, ev):
        # Колесо мыши — масштаб
        delta = ev.angleDelta().y() / 120.0
        factor = 1.1 ** delta
        self.s *= factor
        self.s = max(MIN_SCALE, min(MAX_SCALE, self.s)) # нельзя бесконечно приближать и отдалять
        self.update()

    # Клавиатура
    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            self.reset_transform()
        elif k == QtCore.Qt.Key_Space:
            self.toggle_animation()

        self.update()

    # Анимация
    def start_animation(self):
        # относительно какой позиции анимация
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float) 

        self.anim_running = True
        self.anim_start_time = time.time()

        # инициализация HUD-значений
        self.current_angular_velocity = self.omega0
        self.current_radius = 0.0
        self.update()

    def stop_animation(self):
        self.anim_running = False
        self.anim_start_time = None
        self.update()

    def toggle_animation(self):
        if self.anim_running:
            self.stop_animation()
        else:
            self.start_animation()

    def on_anim_tick(self):
        # Линейное замедление:
        # omega(t) = omega0 * (1 - t/T)  (линейно до 0)
        # theta(t) = интеграл omega = omega0*(t - t^2/(2T))  (для t <= T)
        # радиус растёт линейно: r(t) = R * (t/T)
        # z(t) = z0 + vz * (t/T) при вращение вдоль OZ (аналогично при OX и OY)
        # Авто-стоп при t >= T
        if not self.anim_running:
            return

        t = time.time() - self.anim_start_time
        T = max(self.anim_duration, 1e-6)

        # сколько прошло уже времени
        if t >= T:
            progress = 1.0
            finished = True
        else:
            progress = t / T
            finished = False

        # текущая угловая скорость (линейно уменьшается до 0)
        omega_t = self.omega0 * max(0.0, 1.0 - progress)

        # угол: интеграл omega(t)
        if not finished:
            theta = self.omega0 * (t - (t*t) / (2.0 * T))
        else:
            # если анимация закончилась, угол равен интегралу до T: omega0 * T/2
            theta = 0.5 * self.omega0 * T

        # радиус считаем относительно прошедшего времени
        radius_t = self.radius * progress

        # базовая позиция, с которой начинается анимация
        base = np.array(self._base_pos, dtype=float)

        # В зависимости от выбранной оси строим спираль в соответствующей плоскости
        axis = self.anim_axis.upper()
        if axis == 'OZ':
            # спираль в XY, смещение по Z
            x = base[0] + radius_t * math.cos(theta)
            y = base[1] + radius_t * math.sin(theta)
            z = base[2] + self.vz * progress
            # вращаем вокруг OZ на theta
            self.rz = theta
        elif axis == 'OY':
            # спираль в XZ, смещение по Y
            x = base[0] + radius_t * math.cos(theta)
            z = base[2] + radius_t * math.sin(theta)
            y = base[1] + self.vz * progress
            self.ry = theta
        else:  # 'OX'
            # спираль в YZ, смещение по X
            y = base[1] + radius_t * math.cos(theta)
            z = base[2] + radius_t * math.sin(theta)
            x = base[0] + self.vz * progress
            self.rx = theta

        # обновляем координаты
        self.tx, self.ty, self.tz = float(x), float(y), float(z)

        # обновляем значения
        self.current_angular_velocity = omega_t
        self.current_radius = radius_t

        # автоматически останавливаем по окончании T
        if finished:
            # оставляем финальные значения и останавливаем
            self.stop_animation()

        self.update()

    # сброс до изначальной позиции
    def reset_transform(self):
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0
        self.current_angular_velocity = 0.0
        self.current_radius = 0.0
        self._base_pos = np.array([self.tx, self.ty, self.tz], dtype=float)
        self.update()

# Главное окно
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

        def on_axis_changed(text):
            self.gl.anim_axis = text
            self.gl.update()

        axis_box.currentTextChanged.connect(on_axis_changed)
        layout.addWidget(axis_box)

        layout.addWidget(QtWidgets.QLabel("Начальная угловая скорость (рад/с):"))
        omega_spin = QtWidgets.QDoubleSpinBox()
        omega_spin.setRange(0.1, 40.0); omega_spin.setValue(self.gl.omega0); omega_spin.setSingleStep(0.1)
        omega_spin.valueChanged.connect(lambda v: setattr(self.gl, "omega0", v))
        layout.addWidget(omega_spin)

        layout.addWidget(QtWidgets.QLabel("Целевой радиус спирали:"))
        radius_spin = QtWidgets.QDoubleSpinBox()
        radius_spin.setRange(0.0, 3.0); radius_spin.setValue(self.gl.radius); radius_spin.setSingleStep(0.05)
        radius_spin.valueChanged.connect(lambda v: setattr(self.gl, "radius", v))
        layout.addWidget(radius_spin)

        layout.addWidget(QtWidgets.QLabel("Итоговое смещение:"))
        vz_spin = QtWidgets.QDoubleSpinBox()
        vz_spin.setRange(-5.0, 5.0); vz_spin.setValue(self.gl.vz); vz_spin.setSingleStep(0.05)
        vz_spin.valueChanged.connect(lambda v: setattr(self.gl, "vz", v))
        layout.addWidget(vz_spin)

        layout.addWidget(QtWidgets.QLabel("Длительность анимации (сек):"))
        dur_spin = QtWidgets.QDoubleSpinBox()
        dur_spin.setRange(0.5, 60.0)
        dur_spin.setValue(self.gl.anim_duration)
        dur_spin.setSingleStep(0.5)
        dur_spin.valueChanged.connect(lambda v: setattr(self.gl, "anim_duration", v))
        layout.addWidget(dur_spin)

        btn_row = QtWidgets.QHBoxLayout()
        btn_start = QtWidgets.QPushButton("Старт/Пауза (Space)")
        btn_start.clicked.connect(self.gl.toggle_animation)
        btn_reset = QtWidgets.QPushButton("Сброс (R)")
        btn_reset.clicked.connect(self.gl.reset_transform)
        btn_row.addWidget(btn_start); btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        instr = QtWidgets.QLabel(
            "Управление:\n"
            "- ЛКМ + перетаскивание: вращение (OX/OY)\n"
            "- Shift + ЛКМ: вращение вокруг OZ\n"
            "- ПКМ + перетаскивание: перенос по XY\n"
            "- Shift + ПКМ: перенос вдоль OZ\n"
            "- Колесо мыши / +/- : масштаб\n"
            "- Space: запустить/приостановить анимацию\n"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        layout.addStretch()
        dock = QtWidgets.QDockWidget("Управление", self)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)



# Запуск
app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.resize(1100, 700)
w.show()
sys.exit(app.exec_())