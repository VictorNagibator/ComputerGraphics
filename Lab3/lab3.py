# lab_rev_opengl.py
# Полный файл — заменить старый. Требуется: PyQt5, PyOpenGL, numpy

import sys, math, time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL import GL
from OpenGL.GL import shaders

# GLSL шейдеры для модели освещения по Гуро (Gouraud shading)
# Вершинный шейдер вычисляет освещение для каждой вершины
VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;          // позиция вершины
layout(location = 1) in vec3 aNormal;       // нормаль вершины

uniform mat4 uMVP;                          // матрица модель-вид-проекция
uniform vec3 uLightPosWorld;                // позиция света в мировых координатах
uniform float uKa;                          // коэффициент окружающего освещения
uniform float uKd;                          // коэффициент диффузного освещения  
uniform float uIa;                          // интенсивность окружающего освещения
uniform float uIl;                          // интенсивность источника света

out float vIntensity;                       // выходная интенсивность освещения

void main() {
    vec4 posWorld = vec4(aPos, 1.0);        // позиция в мировых координатах
    vec3 N = normalize(aNormal);             // нормализуем нормаль
    vec3 L = normalize(uLightPosWorld - posWorld.xyz); // направление к свету
    
    float diff = max(0.0, dot(N, L));       // диффузная составляющая
    float I = uIa * uKa + uIl * uKd * diff; // общая интенсивность
    vIntensity = clamp(I, 0.0, 1.0);        // ограничиваем диапазон
    
    gl_Position = uMVP * vec4(aPos, 1.0);   // преобразуем позицию
}
"""

# Фрагментный шейдер - просто использует вычисленную интенсивность
FRAG_SHADER = """
#version 330 core
in float vIntensity;                        // интенсивность из вершинного шейдера
out vec4 FragColor;                         // выходной цвет

void main() {
    float v = clamp(vIntensity, 0.0, 1.0);  // ограничиваем интенсивность
    FragColor = vec4(v, v, v, 1.0);         // grayscale цвет
}
"""

# Шейдеры для осей (простой цвет)
AXIS_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform vec3 uColor;
out vec3 vColor;
void main(){ 
    vColor = uColor; 
    gl_Position = uMVP * vec4(aPos,1.0); 
}
"""

AXIS_FRAG = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){ 
    FragColor = vec4(vColor,1.0); 
}
"""

# Матричные преобразования
def perspective(fovy, aspect, znear, zfar):
    """Создает матрицу перспективной проекции"""
    f = 1.0 / math.tan(fovy / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect                    # масштабирование по X
    M[1,1] = f                             # масштабирование по Y
    M[2,2] = (zfar + znear) / (znear - zfar) # преобразование Z
    M[2,3] = (2.0 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0                          # перспективное деление
    return M

def look_at(eye, center, up):
    """Создает матрицу вида (камеры)"""
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # Вычисляем базисные векторы камеры
    f = center - eye
    f = f / np.linalg.norm(f)               # направление вперед
    u = up / np.linalg.norm(up)             # вверх
    s = np.cross(f, u)                      # право
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)                      # пересчитываем вверх
    
    # Матрица вращения и平移
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s
    M[1,0:3] = u  
    M[2,0:3] = -f
    
    T = np.eye(4, dtype=np.float32)
    T[0:3,3] = -eye                         # перенос
    
    return M @ T                            # вращение * перенос

# Генераторы геометрии
def make_sphere_raw(radius=1.0, lat_segments=20, lon_segments=36):
    """Генерирует сферу как треугольную сетку"""
    positions = []
    indices = []
    lat_segments = max(2, int(lat_segments))
    lon_segments = max(3, int(lon_segments))
    
    # Генерируем вершины
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments    # угол от 0 до π
        y = radius * math.cos(phi)          # координата Y
        r = radius * math.sin(phi)          # радиус на этой широте
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments  # угол от 0 до 2π
            x = r * math.cos(theta)         # координата X
            z = r * math.sin(theta)         # координата Z
            positions.append((x,y,z))
    
    # Генерируем индексы треугольников
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j                   # текущая вершина
            b = (i + 1) * L + j             # вершина ниже
            c = (i + 1) * L + ((j + 1) % L) # вершина ниже и справа
            d = i * L + ((j + 1) % L)       # вершина справа
            
            # Два треугольника образуют квад
            indices.extend((a,b,c))
            indices.extend((a,c,d))
            
    return np.array(positions, dtype=np.float32), np.array(indices, dtype=np.uint32)

def make_ellipsoid_raw(rx=1.4, ry=0.8, rz=1.0, lat_segments=18, lon_segments=32):
    """Генерирует эллипсоид как треугольную сетку"""
    pos = []
    idx = []
    lat_segments = max(2, int(lat_segments))
    lon_segments = max(3, int(lon_segments))
    
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments
        cy = math.cos(phi)
        sy = math.sin(phi)
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            cx = math.cos(theta)
            sx = math.sin(theta)
            # Координаты с учетом разных радиусов по осям
            x = rx * sy * cx
            y = ry * cy  
            z = rz * sy * sx
            pos.append((x,y,z))
    
    # Триангуляция аналогична сфере
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j
            b = (i + 1) * L + j
            c = (i + 1) * L + ((j + 1) % L)
            d = i * L + ((j + 1) % L)
            idx.extend((a,b,c))
            idx.extend((a,c,d))
            
    return np.array(pos,dtype=np.float32), np.array(idx,dtype=np.uint32)

def make_torus_raw_with_normals(R=1.0, r=0.35, seg_major=36, seg_minor=18):
    """Генерирует тор с нормалями"""
    pos = []
    norms = []
    idx = []
    seg_major = max(3, int(seg_major))
    seg_minor = max(3, int(seg_minor))
    
    for i in range(seg_major):
        u = 2.0 * math.pi * i / seg_major   # угол основного кольца
        cu = math.cos(u)
        su = math.sin(u)
        for j in range(seg_minor):
            v = 2.0 * math.pi * j / seg_minor # угол поперечного сечения
            cv = math.cos(v)
            sv = math.sin(v)
            
            # Параметрические уравнения тора
            x = (R + r * cv) * cu
            y = r * sv
            z = (R + r * cv) * su
            pos.append((x,y,z))
            
            # Аналитические нормали для тора
            nx = cu * cv
            ny = sv  
            nz = su * cv
            n = np.array((nx,ny,nz), dtype=np.float32)
            n /= (np.linalg.norm(n) + 1e-12)  # нормализуем
            norms.append(tuple(n))
    
    # Триангуляция - ИСПРАВЛЕНИЕ: создаем обе стороны тора
    M = seg_minor
    for i in range(seg_major):
        for j in range(seg_minor):
            a = i * M + j
            b = ((i + 1) % seg_major) * M + j           # следующее основное кольцо
            c = ((i + 1) % seg_major) * M + ((j + 1) % M) # следующее кольцо + угол
            d = i * M + ((j + 1) % M)                  # следующий угол
            
            # Внешняя сторона
            idx.extend((a,b,c))
            idx.extend((a,c,d))
            
            # Внутренняя сторона - ИСПРАВЛЕНИЕ: добавляем обратную сторону
            idx.extend((c,b,a))
            idx.extend((d,c,a))
            
    return np.array(pos,dtype=np.float32), np.array(norms,dtype=np.float32), np.array(idx,dtype=np.uint32)

# Вычисление нормалей и ориентация
def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray):
    """Вычисляет нормали вершин как среднее нормалей смежных треугольников"""
    N = positions.shape[0]
    normals = np.zeros((N,3), dtype=np.float64)  # используем float64 для точности
    
    # Суммируем нормали всех смежных треугольников
    tri_count = indices.size // 3
    for t in range(tri_count):
        i0 = int(indices[3*t+0])
        i1 = int(indices[3*t+1]) 
        i2 = int(indices[3*t+2])
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        
        # Нормаль треугольника
        v1 = p1 - p0
        v2 = p2 - p0
        fn = np.cross(v1, v2)
        nlen = np.linalg.norm(fn)
        if nlen > 1e-12:
            fn = fn / nlen                   # нормализуем
        else:
            fn = np.array([0.0,0.0,0.0])    # вырожденный треугольник
            
        # Добавляем к вершинам
        normals[i0] += fn
        normals[i1] += fn  
        normals[i2] += fn
    
    # Нормализуем результирующие нормали
    for i in range(N):
        l = np.linalg.norm(normals[i])
        if l < 1e-12:
            normals[i] = np.array([0.0,0.0,1.0])  # нормаль по умолчанию
        else:
            normals[i] /= l
            
    return normals.astype(np.float32)        # возвращаем как float32 для OpenGL

def orient_triangles_outward(positions: np.ndarray, indices: np.ndarray):
    """Ориентирует треугольники наружу для корректного backface culling"""
    idx = indices.copy().astype(np.int64)
    
    for t in range(idx.size // 3):
        i0 = idx[3*t+0]
        i1 = idx[3*t+1] 
        i2 = idx[3*t+2]
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        
        # Нормаль треугольника
        fn = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(fn) < 1e-12:
            continue                         # пропускаем вырожденные
            
        centroid = (p0 + p1 + p2) / 3.0     # центр треугольника
        
        # Если нормаль направлена к центру объекта - меняем порядок вершин
        if np.dot(fn, centroid) < 0.0:
            idx[3*t+1], idx[3*t+2] = idx[3*t+2], idx[3*t+1]  # меняем порядок
            
    return idx.astype(np.uint32)

# Основной виджет OpenGL
class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Настройка формата OpenGL контекста
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)           # буфер глубины 24 бита
        fmt.setProfile(QSurfaceFormat.CoreProfile)  # core profile
        fmt.setVersion(3, 3)                 # OpenGL 3.3
        self.setFormat(fmt)

        # Параметры модели
        self.shape = 'sphere'                # текущая форма
        self.lat_segments = 24               # сегменты по широте
        self.lon_segments = 40               # сегменты по долготе

        # Камера (сферические координаты)
        self.camera_target = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.cam_radius = 4.5                # расстояние до цели
        self.cam_az = 0.0                    # азимут
        self.cam_el = math.radians(10.0)     # угол возвышения
        self.camera_up = np.array([0.0,1.0,0.0], dtype=np.float32)

        # Освещение
        self.light_world = np.array([2.5,3.0,2.5], dtype=np.float32)
        self.light_on = True
        self.Ka = 0.35                       # коэффициент ambient
        self.Kd = 1.0                        # коэффициент diffuse  
        self.Ia = 0.25                       # интенсивность ambient
        self.Il = 1.2                        # интенсивность источника

        # Переключатели отображения
        self.show_wireframe = True           # показывать каркас
        self.show_normals = False            # показывать нормали
        self.show_axes = True                # показывать оси

        # Анимация движения камеры - ИСПРАВЛЕНИЕ: улучшенная анимация
        self.cam_move_active = False         # активно ли движение
        self.cam_move_start = 0.0            # время начала
        self.cam_move_dur = 1500             # длительность в ms
        self.cam_move_from_az = 0.0          # начальный азимут
        self.cam_move_from_el = 0.0          # начальный угол возвышения
        self.cam_move_from_radius = 0.0      # начальный радиус

        # Ресурсы OpenGL
        self.program = None                  # основная программа шейдеров
        self.axis_program = None             # программа для осей
        self.use_vao = True                  # использовать VAO
        self.vao = None                      # Vertex Array Object
        self.vbo = None                      # Vertex Buffer Object  
        self.ebo = None                      # Element Buffer Object
        self.index_count = 0                 # количество индексов
        
        self.axes_vao = None                 # VAO для осей
        self.axes_vbo = None                 # VBO для осей
        
        self.light_vao = None                # VAO для маркера света
        self.light_vbo = None                # VBO для маркера света
        
        self.normals_vao = None              # VAO для нормалей
        self.normals_vbo = None              # VBO для нормалей
        self.normals_count = 0               # количество вершин нормалей

        # Ввод
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.mouse_mode = None               # режим мыши
        self.last_mouse = None               # последняя позиция мыши

        # Таймер для анимации
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)                 # ~60 FPS

    def initializeGL(self):
        """Инициализация OpenGL"""
        try:
            # Компилируем шейдеры
            vert = shaders.compileShader(VERT_SHADER, GL.GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAG_SHADER, GL.GL_FRAGMENT_SHADER)
            self.program = shaders.compileProgram(vert, frag)
            
            vert2 = shaders.compileShader(AXIS_VERT, GL.GL_VERTEX_SHADER)
            frag2 = shaders.compileShader(AXIS_FRAG, GL.GL_FRAGMENT_SHADER)
            self.axis_program = shaders.compileProgram(vert2, frag2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Shader error", str(e))
            self.program = None
            return

        # Проверяем поддержку VAO
        try:
            vao_test = GL.glGenVertexArrays(1)
            try: 
                GL.glDeleteVertexArrays(1, [vao_test])
            except Exception: 
                pass
            self.use_vao = True
        except Exception:
            print("VAO not available, fallback.")
            self.use_vao = False

        # Создаем геометрию
        self.create_mesh()
        self.create_axes()
        self.create_light_marker()
        self.create_normals_vbo()  # создается после создания меша

        # Настройки OpenGL
        GL.glEnable(GL.GL_DEPTH_TEST)        # тест глубины
        try:
            GL.glEnable(GL.GL_CULL_FACE)     # отсечение задних граней
            GL.glCullFace(GL.GL_BACK)        # отсекать задние
            GL.glFrontFace(GL.GL_CCW)        # лицевые - против часовой
        except Exception:
            pass
            
        GL.glClearColor(0.96,0.96,0.96,1.0) # цвет фона

    def create_mesh(self):
        """Создает меш в зависимости от выбранной формы"""
        if self.shape == 'sphere':
            pos, idx = make_sphere_raw(radius=1.0, lat_segments=self.lat_segments, lon_segments=self.lon_segments)
            # Аналитические нормали для сферы - просто нормализованные позиции
            normals = np.zeros_like(pos, dtype=np.float32)
            for i in range(pos.shape[0]):
                p = pos[i]
                n = p / (np.linalg.norm(p) + 1e-12)
                normals[i] = n
                
        elif self.shape == 'ellipsoid':
            pos, idx = make_ellipsoid_raw(rx=1.4, ry=0.8, rz=1.0, 
                                        lat_segments=max(4,self.lat_segments//1), 
                                        lon_segments=self.lon_segments)
            # Аналитические нормали для эллипсоида
            rx, ry, rz = 1.4, 0.8, 1.0
            normals = np.zeros_like(pos, dtype=np.float32)
            for i in range(pos.shape[0]):
                x,y,z = pos[i]
                nx = x/(rx*rx)  # нормаль к эллипсоиду
                ny = y/(ry*ry)
                nz = z/(rz*rz)
                n = np.array((nx,ny,nz), dtype=np.float32)
                n /= (np.linalg.norm(n) + 1e-12)
                normals[i] = n
                
        elif self.shape == 'torus':
            pos, analytic_norms, idx = make_torus_raw_with_normals(R=1.0, r=0.4, 
                                                                seg_major=self.lon_segments, 
                                                                seg_minor=max(8,self.lat_segments//2))
            normals = analytic_norms.astype(np.float32)  # используем аналитические нормали
            
            # ИСПРАВЛЕНИЕ: для тора отключаем backface culling чтобы видеть внутреннюю часть
            try:
                GL.glDisable(GL.GL_CULL_FACE)
            except Exception:
                pass
        else:
            # Fallback - сфера
            pos, idx = make_sphere_raw()
            normals = compute_vertex_normals(pos, idx)

        # Ориентируем треугольники наружу для корректного backface culling
        try:
            idx = orient_triangles_outward(pos, idx)
            
            # Определяем глобальное направление обхода
            try:
                p0 = pos[idx[0::3]]  # первые вершины всех треугольников
                p1 = pos[idx[1::3]]  # вторые вершины
                p2 = pos[idx[2::3]]  # третьи вершины
                
                fn = np.cross(p1 - p0, p2 - p0)  # нормали треугольников
                cent = (p0 + p1 + p2) * (1.0/3.0)  # центроиды
                dots = (fn * cent).sum(axis=1)     # скалярные произведения
                score = np.mean(np.sign(dots))     # среднее направление
                
                if score < 0.0:
                    GL.glFrontFace(GL.GL_CW)       # по часовой если внутрь
                else:
                    GL.glFrontFace(GL.GL_CCW)      # против часовой если наружу
            except Exception:
                GL.glFrontFace(GL.GL_CCW)          # по умолчанию
        except Exception:
            pass

        # Пересчитываем нормали для гарантии сглаживания (Gouraud)
        normals = compute_vertex_normals(pos, idx)

        # Подготавливаем массивы для OpenGL
        pos = np.ascontiguousarray(pos, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        idx = np.ascontiguousarray(idx, dtype=np.uint32)

        self.index_count = idx.size
        
        # Интерливинг позиций и нормалей: [x,y,z,nx,ny,nz, x,y,z,nx,ny,nz, ...]
        verts_interleaved = np.ascontiguousarray(np.hstack((pos, normals)).astype(np.float32), dtype=np.float32)

        # Удаляем старые буферы если есть
        try:
            if getattr(self, 'vao', None) and self.use_vao:
                GL.glDeleteVertexArrays(1, [self.vao])
        except Exception: 
            pass
        try:
            if getattr(self, 'vbo', None): 
                GL.glDeleteBuffers(1, [self.vbo])
        except Exception: 
            pass
        try:
            if getattr(self, 'ebo', None): 
                GL.glDeleteBuffers(1, [self.ebo])
        except Exception: 
            pass

        # Создаем VBO и EBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_interleaved.nbytes, verts_interleaved, GL.GL_STATIC_DRAW)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

        # Создаем VAO если поддерживается
        if self.use_vao:
            try:
                self.vao = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(self.vao)
                
                # Привязываем буферы
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                
                # Настраиваем атрибуты вершин
                # Атрибут 0: позиции (3 float)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                
                # Атрибут 1: нормали (3 float, смещение 3*4 байт)
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
                
                GL.glBindVertexArray(0)  # отвязываем VAO
            except Exception as e:
                print("VAO creation failed, disabling VAO:", e)
                self.use_vao = False
                try: 
                    GL.glDeleteVertexArrays(1, [self.vao])
                except Exception: 
                    pass
                self.vao = None
        else:
            self.vao = None

        # Создаем VBO для отображения нормалей (линии)
        self._build_normals_lines(pos, normals)

        # Отвязываем буферы
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _build_normals_lines(self, pos, normals, scale=0.12):
        """Создает линии для визуализации нормалей"""
        lines = []
        for i in range(pos.shape[0]):
            p = pos[i]                       # начальная точка
            n = normals[i]                   # направление нормали
            lines.append(p.tolist())         # от вершины
            lines.append((p + n * scale).tolist())  # до вершины + нормаль*масштаб
            
        arr = np.array(lines, dtype=np.float32)
        self.normals_count = arr.shape[0]    # количество вершин для линий
        
        # Удаляем старые буферы
        try:
            if getattr(self, 'normals_vao', None): 
                GL.glDeleteVertexArrays(1, [self.normals_vao])
            if getattr(self, 'normals_vbo', None): 
                GL.glDeleteBuffers(1, [self.normals_vbo])
        except Exception:
            pass
            
        # Создаем новые буферы
        self.normals_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
        
        if self.use_vao:
            self.normals_vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.normals_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        else:
            self.normals_vao = None
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def create_axes(self):
        """Создает оси координат"""
        # Вершины для осей: X, Y, Z
        axes = np.array([
            0,0,0, 1.5,0,0,  # ось X
            0,0,0, 0,1.5,0,  # ось Y  
            0,0,0, 0,0,1.5   # ось Z
        ], dtype=np.float32)
        
        try:
            if getattr(self, 'axes_vao', None): 
                GL.glDeleteVertexArrays(1, [self.axes_vao])
            if getattr(self, 'axes_vbo', None): 
                GL.glDeleteBuffers(1, [self.axes_vbo])
        except Exception:
            pass
            
        self.axes_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, axes.nbytes, axes, GL.GL_STATIC_DRAW)
        
        if self.use_vao:
            self.axes_vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.axes_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        else:
            self.axes_vao = None
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def create_light_marker(self):
        """Создает маркер для источника света"""
        try:
            if getattr(self, 'light_vao', None): 
                GL.glDeleteVertexArrays(1, [self.light_vao])
            if getattr(self, 'light_vbo', None): 
                GL.glDeleteBuffers(1, [self.light_vbo])
        except Exception:
            pass
            
        self.light_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 3*4, None, GL.GL_DYNAMIC_DRAW)  # 3 float
        
        if self.use_vao:
            self.light_vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.light_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        else:
            self.light_vao = None
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def create_normals_vbo(self):
        """Заглушка - реальный буфер создается в create_mesh"""
        self.normals_vao = None
        self.normals_vbo = None
        self.normals_count = 0

    def resizeGL(self, w, h):
        """Обработка изменения размера окна"""
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        """Основная функция отрисовки"""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.program:
            return

        # Вычисляем матрицы
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / float(h)
        
        proj = perspective(math.radians(60.0), aspect, 0.1, 100.0)  # проекция
        
        eye = self._sph_to_cart(self.cam_az, self.cam_el, self.cam_radius) + self.camera_target
        view = look_at(eye, self.camera_target, self.camera_up)      # вид
        mv = view @ np.eye(4, dtype=np.float32)                      # модель-вид
        mvp = proj @ mv                                              # итоговая матрица

        # Отрисовка основной модели
        GL.glUseProgram(self.program)
        
        # Передаем uniform-переменные
        loc = GL.glGetUniformLocation(self.program, 'uMVP')
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)  # транспонируем для OpenGL
        
        loc = GL.glGetUniformLocation(self.program, 'uLightPosWorld')
        GL.glUniform3f(loc, float(self.light_world[0]), float(self.light_world[1]), float(self.light_world[2]))
        
        # Параметры освещения
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKa'), float(self.Ka))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKd'), float(self.Kd))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIa'), float(self.Ia))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIl'), float(self.Il) if self.light_on else 0.0)

        # ИСПРАВЛЕНИЕ: для тора временно отключаем backface culling
        cull_was_enabled = GL.glIsEnabled(GL.GL_CULL_FACE)
        if self.shape == 'torus':
            GL.glDisable(GL.GL_CULL_FACE)
        else:
            if not cull_was_enabled:
                GL.glEnable(GL.GL_CULL_FACE)

        # Отрисовка заполненных треугольников со смещением
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)  # смещение чтобы избежать z-fighting
        
        if self.use_vao and self.vao is not None:
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
        else:
            # Fallback без VAO
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        
        # Восстанавливаем состояние backface culling
        if self.shape == 'torus' and cull_was_enabled:
            GL.glEnable(GL.GL_CULL_FACE)
        elif self.shape != 'torus' and not cull_was_enabled:
            GL.glDisable(GL.GL_CULL_FACE)
            
        GL.glUseProgram(0)

        # Каркасная модель поверх
        if self.show_wireframe:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 0.08, 0.08, 0.08)  # темно-серый
            
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glLineWidth(1.0)
            
            if self.use_vao and self.vao is not None:
                GL.glBindVertexArray(self.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
                
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glUseProgram(0)

        # Визуализация нормалей
        if self.show_normals and getattr(self, 'normals_count', 0) > 0:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 0.0, 0.45, 0.8)  # синий цвет
            
            if self.use_vao and self.normals_vao is not None:
                GL.glBindVertexArray(self.normals_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, self.normals_count)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_LINES, 0, self.normals_count)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
            GL.glUseProgram(0)

        # Оси координат
        if self.show_axes:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            
            # Ось X - красная
            GL.glUniform3f(locc, 1.0, 0.0, 0.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glBindVertexArray(self.axes_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glDisableVertexAttribArray(0)
                
            # Ось Y - зеленая
            GL.glUniform3f(locc, 0.0, 1.0, 0.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glDrawArrays(GL.GL_LINES, 2, 2)
            else:
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(2*3*4))
                GL.glDrawArrays(GL.GL_LINES, 0, 2)  # рисуем из того же буфера но со смещением
                
            # Ось Z - синяя
            GL.glUniform3f(locc, 0.0, 0.0, 1.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glDrawArrays(GL.GL_LINES, 4, 2)
                GL.glBindVertexArray(0)
            else:
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(4*3*4))
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
            GL.glUseProgram(0)

        # Маркер источника света
        if self.light_vbo is not None:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 1.0, 0.85, 0.0)  # желтый цвет
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
            data = np.array(self.light_world, dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)  # обновляем позицию
            
            if self.use_vao and self.light_vao is not None:
                GL.glBindVertexArray(self.light_vao)
                GL.glPointSize(8.0)  # размер точки
                GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glBindVertexArray(0)
            else:
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glPointSize(8.0)
                GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glDisableVertexAttribArray(0)
                
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

    def _sph_to_cart(self, az, el, r):
        """Преобразует сферические координаты в декартовы"""
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        return np.array([x, y, z], dtype=np.float32)

    # Обработка ввода
    def mousePressEvent(self, ev):
        """Обработка нажатия мыши"""
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouse_mode = 'orbit'
            self._orbit_start = (self.cam_az, self.cam_el, self.cam_radius)  # запоминаем начальное состояние
            self._orbit_mouse = ev.pos()
        self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        """Обработка движения мыши"""
        if self.last_mouse is None:
            return
            
        if self.mouse_mode == 'orbit':
            az0, el0, r0 = self._orbit_start
            ddx = ev.x() - self._orbit_mouse.x()
            ddy = ev.y() - self._orbit_mouse.y()
            
            az = az0 + ddx * 0.01  # чувствительность
            el = el0 + ddy * 0.01
            
            # Ограничиваем угол возвышения
            max_el = math.radians(89.0)
            el = max(-max_el, min(max_el, el))
            
            self.cam_az = az
            self.cam_el = el
            self.update()
            
        self.last_mouse = ev.pos()

    def mouseReleaseEvent(self, ev):
        """Обработка отпускания мыши"""
        self.mouse_mode = None
        self.last_mouse = None

    def keyPressEvent(self, ev):
        """Обработка нажатия клавиш"""
        if ev.key() == QtCore.Qt.Key_Space:
            # Вкл/выкл автоматическое вращение
            self.anim_running = not getattr(self, 'anim_running', False)
            self.update()
        else:
            super().keyPressEvent(ev)

    # Анимация движения камеры - ИСПРАВЛЕНИЕ: улучшенная плавная анимация
    def start_camera_move_via_light(self, duration_ms=1500):
        """Запускает анимацию движения камеры через источник света"""
        # Запоминаем начальные параметры камеры
        self.cam_move_from_az = self.cam_az
        self.cam_move_from_el = self.cam_el
        self.cam_move_from_radius = self.cam_radius
        
        # Вычисляем позицию света в сферических координатах
        light_vec = self.light_world - self.camera_target
        light_r = np.linalg.norm(light_vec)
        light_el = math.asin(light_vec[1] / light_r) if light_r > 1e-6 else 0.0
        light_az = math.atan2(light_vec[2], light_vec[0])
        
        # Вычисляем противоположную позицию
        opposite_az = light_az + math.pi  # противоположный азимут
        opposite_el = -light_el           # противоположный угол возвышения
        
        self.cam_move_to_az = opposite_az
        self.cam_move_to_el = opposite_el
        self.cam_move_to_radius = light_r  # сохраняем радиус
        
        self.cam_move_start = time.time() * 1000.0
        self.cam_move_dur = max(50, int(duration_ms))
        self.cam_move_active = True

    def on_timer(self):
        """Обработка таймера для анимации"""
        now = time.time() * 1000.0
        
        # Автоматическое вращение
        if getattr(self, 'anim_running', False):
            self.cam_az += 0.01
            self.update()
            
        # Анимация движения камеры - ИСПРАВЛЕНИЕ: плавная интерполяция
        if getattr(self, 'cam_move_active', False):
            elapsed = now - self.cam_move_start
            t = min(1.0, elapsed / float(self.cam_move_dur))
            
            # Плавная интерполяция с ease-in-out
            smooth_t = self._ease_in_out(t)
            
            # Интерполируем параметры камеры
            self.cam_az = self._lerp_angle(self.cam_move_from_az, self.cam_move_to_az, smooth_t)
            self.cam_el = self._lerp(self.cam_move_from_el, self.cam_move_to_el, smooth_t)
            self.cam_radius = self._lerp(self.cam_move_from_radius, self.cam_move_to_radius, smooth_t)
            
            if t >= 1.0:
                self.cam_move_active = False  # завершаем анимацию
                
            self.update()

    def _ease_in_out(self, t):
        """Функция плавности для анимации"""
        return 0.5 * (1 - math.cos(t * math.pi))

    def _lerp(self, a, b, t):
        """Линейная интерполяция"""
        return a + (b - a) * t

    def _lerp_angle(self, a, b, t):
        """Интерполяция углов с учетом циклической природы"""
        # Нормализуем углы к диапазону [-π, π]
        a = self._normalize_angle(a)
        b = self._normalize_angle(b)
        
        # Выбираем кратчайший путь
        diff = b - a
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
            
        return a + diff * t

    def _normalize_angle(self, angle):
        """Нормализует угол к диапазону [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# Главное окно с русским интерфейсом
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Лабораторная: Шар + OpenGL (Gouraud)')
        self.gl = GLWidget()
        self.setCentralWidget(self.gl)
        
        self.create_controls()
        self.resize(1300, 820)
        
        # Таймер для обновления UI
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10 FPS для UI

    def create_controls(self):
        """Создает панель управления"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel('<b>Управление сценой</b>'))

        # Выбор формы
        shape_h = QtWidgets.QHBoxLayout()
        btn_sphere = QtWidgets.QPushButton('Сфера')
        btn_sphere.clicked.connect(lambda: self.set_shape('sphere'))
        btn_ell = QtWidgets.QPushButton('Эллипсоид')
        btn_ell.clicked.connect(lambda: self.set_shape('ellipsoid'))
        btn_tor = QtWidgets.QPushButton('Тор')
        btn_tor.clicked.connect(lambda: self.set_shape('torus'))
        
        shape_h.addWidget(btn_sphere)
        shape_h.addWidget(btn_ell)
        shape_h.addWidget(btn_tor)
        layout.addLayout(shape_h)

        # Настройки сетки
        layout.addWidget(QtWidgets.QLabel('Настройки сетки (lat x lon)'))
        grid_h = QtWidgets.QHBoxLayout()
        self.lat_spin = QtWidgets.QSpinBox()
        self.lat_spin.setRange(3, 160)
        self.lat_spin.setValue(self.gl.lat_segments)
        self.lon_spin = QtWidgets.QSpinBox()
        self.lon_spin.setRange(3, 320)
        self.lon_spin.setValue(self.gl.lon_segments)
        
        self.lat_spin.valueChanged.connect(self.on_mesh_changed)
        self.lon_spin.valueChanged.connect(self.on_mesh_changed)
        
        grid_h.addWidget(self.lat_spin)
        grid_h.addWidget(self.lon_spin)
        layout.addLayout(grid_h)

        # Переключатели отображения
        self.tri_check = QtWidgets.QCheckBox('Показывать триангуляцию (wireframe)')
        self.tri_check.setChecked(self.gl.show_wireframe)
        self.tri_check.stateChanged.connect(lambda s: self.on_wire_toggle(s))
        
        self.norm_check = QtWidgets.QCheckBox('Показывать векторы нормалей (по вершинам)')
        self.norm_check.setChecked(self.gl.show_normals)
        self.norm_check.stateChanged.connect(lambda s: self.on_normals_toggle(s))
        
        self.axes_check = QtWidgets.QCheckBox('Показывать оси')
        self.axes_check.setChecked(self.gl.show_axes)
        self.axes_check.stateChanged.connect(lambda s: self.on_axes_toggle(s))
        
        layout.addWidget(self.tri_check)
        layout.addWidget(self.norm_check)
        layout.addWidget(self.axes_check)

        # Освещение
        layout.addWidget(QtWidgets.QLabel('<b>Свет</b>'))
        self.light_check = QtWidgets.QCheckBox('Включить свет')
        self.light_check.setChecked(True)
        self.light_check.stateChanged.connect(self.on_light_toggle)
        layout.addWidget(self.light_check)
        
        # Координаты света
        coords_h = QtWidgets.QHBoxLayout()
        self.lx = QtWidgets.QDoubleSpinBox()
        self.lx.setRange(-20.0, 20.0)
        self.lx.setSingleStep(0.1)
        self.lx.setValue(float(self.gl.light_world[0]))
        self.lx.valueChanged.connect(self.on_light_changed)
        
        self.ly = QtWidgets.QDoubleSpinBox()
        self.ly.setRange(-20.0, 20.0)
        self.ly.setSingleStep(0.1)
        self.ly.setValue(float(self.gl.light_world[1]))
        self.ly.valueChanged.connect(self.on_light_changed)
        
        self.lz = QtWidgets.QDoubleSpinBox()
        self.lz.setRange(-20.0, 20.0)
        self.lz.setSingleStep(0.1)
        self.lz.setValue(float(self.gl.light_world[2]))
        self.lz.valueChanged.connect(self.on_light_changed)
        
        coords_h.addWidget(QtWidgets.QLabel('x'))
        coords_h.addWidget(self.lx)
        coords_h.addWidget(QtWidgets.QLabel('y'))
        coords_h.addWidget(self.ly)
        coords_h.addWidget(QtWidgets.QLabel('z'))
        coords_h.addWidget(self.lz)
        layout.addLayout(coords_h)

        # Параметры освещения
        params_h = QtWidgets.QGridLayout()
        params_h.addWidget(QtWidgets.QLabel('Ka (ambient)'), 0, 0)
        self.ka_spin = QtWidgets.QDoubleSpinBox()
        self.ka_spin.setRange(0.0, 2.0)
        self.ka_spin.setSingleStep(0.05)
        self.ka_spin.setValue(self.gl.Ka)
        self.ka_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ka_spin, 0, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Kd (diffuse)'), 1, 0)
        self.kd_spin = QtWidgets.QDoubleSpinBox()
        self.kd_spin.setRange(0.0, 2.0)
        self.kd_spin.setSingleStep(0.05)
        self.kd_spin.setValue(self.gl.Kd)
        self.kd_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.kd_spin, 1, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Ia (ambient intensity)'), 2, 0)
        self.ia_spin = QtWidgets.QDoubleSpinBox()
        self.ia_spin.setRange(0.0, 2.0)
        self.ia_spin.setSingleStep(0.05)
        self.ia_spin.setValue(self.gl.Ia)
        self.ia_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ia_spin, 2, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Il (light intensity)'), 3, 0)
        self.il_spin = QtWidgets.QDoubleSpinBox()
        self.il_spin.setRange(0.0, 5.0)
        self.il_spin.setSingleStep(0.05)
        self.il_spin.setValue(self.gl.Il)
        self.il_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.il_spin, 3, 1)
        
        layout.addLayout(params_h)

        # Кнопка движения камеры
        move_h = QtWidgets.QHBoxLayout()
        self.btn_move = QtWidgets.QPushButton('Движение: свет → противоположная (дуга)')
        self.btn_move.clicked.connect(self.on_start_move)
        move_h.addWidget(self.btn_move)
        layout.addLayout(move_h)

        # Скорость анимации
        sp_h = QtWidgets.QHBoxLayout()
        sp_h.addWidget(QtWidgets.QLabel('Скорость движения (ms)'))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(200, 5000)
        self.speed_slider.setValue(1500)
        self.speed_label = QtWidgets.QLabel('1500')
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        
        sp_h.addWidget(self.speed_slider)
        sp_h.addWidget(self.speed_label)
        layout.addLayout(sp_h)

        # Инструкция
        instr = QtWidgets.QLabel('Space — авто-вращение (az). Орбита мышью: ЛКМ+перетаскивание (вращение вокруг цели).')
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Отображение координат
        layout.addWidget(QtWidgets.QLabel('<b>Камера (eye)</b>'))
        self.camera_label = QtWidgets.QLabel('x: 0.00, y: 0.00, z: 0.00')
        layout.addWidget(self.camera_label)
        
        layout.addWidget(QtWidgets.QLabel('<b>Источник света (координаты)</b>'))
        self.light_label = QtWidgets.QLabel('x: 0.00, y: 0.00, z: 0.00')
        layout.addWidget(self.light_label)

        layout.addStretch()
        panel.setLayout(layout)
        
        # Создаем док-виджет для панели управления
        dock = QtWidgets.QDockWidget('Панель управления', self)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # Обработчики UI
    def set_shape(self, s):
        """Устанавливает форму объекта"""
        self.gl.makeCurrent()
        self.gl.shape = s
        self.gl.create_mesh()  # пересоздаем геометрию
        self.gl.doneCurrent()
        self.gl.update()

    def on_mesh_changed(self, val=None):
        """Обработчик изменения параметров сетки"""
        try:
            self.gl.makeCurrent()
            self.gl.lat_segments = int(self.lat_spin.value())
            self.gl.lon_segments = int(self.lon_spin.value())
            self.gl.create_mesh()  # пересоздаем с новыми параметрами
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    def on_wire_toggle(self, state):
        """Переключатель каркасного режима"""
        self.gl.show_wireframe = (state == QtCore.Qt.Checked)
        self.gl.update()

    def on_normals_toggle(self, state):
        """Переключатель отображения нормалей"""
        self.gl.show_normals = (state == QtCore.Qt.Checked)
        self.gl.update()

    def on_axes_toggle(self, state):
        """Переключатель отображения осей"""
        self.gl.show_axes = (state == QtCore.Qt.Checked)
        self.gl.update()

    def on_light_toggle(self, state):
        """Переключатель освещения"""
        self.gl.light_on = (state == QtCore.Qt.Checked)
        self.gl.update()

    def on_light_changed(self, val=None):
        """Обработчик изменения позиции света"""
        try:
            self.gl.makeCurrent()
            self.gl.light_world = np.array([float(self.lx.value()), float(self.ly.value()), float(self.lz.value())], dtype=np.float32)
            self.gl.create_light_marker()  # обновляем маркер
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    def on_light_params_changed(self, val=None):
        """Обработчик изменения параметров освещения"""
        self.gl.Ka = float(self.ka_spin.value())
        self.gl.Kd = float(self.kd_spin.value())
        self.gl.Ia = float(self.ia_spin.value())
        self.gl.Il = float(self.il_spin.value())
        self.gl.update()

    def on_start_move(self):
        """Запуск анимации движения камеры"""
        dur = int(self.speed_slider.value())
        self.gl.start_camera_move_via_light(duration_ms=dur)
        self.gl.update()

    def update_ui(self):
        """Обновление UI (вызывается таймером)"""
        # Обновляем координаты света
        lw = self.gl.light_world
        self.light_label.setText(f'x: {lw[0]:.2f}, y: {lw[1]:.2f}, z: {lw[2]:.2f}')
        
        # Обновляем координаты камеры
        eye = self.gl._sph_to_cart(self.gl.cam_az, self.gl.cam_el, self.gl.cam_radius) + self.gl.camera_target
        self.camera_label.setText(f'x: {eye[0]:.2f}, y: {eye[1]:.2f}, z: {eye[2]:.2f}')
        
        # Синхронизируем состояния переключателей
        if self.tri_check.isChecked() != self.gl.show_wireframe:
            self.tri_check.setChecked(self.gl.show_wireframe)
        if self.norm_check.isChecked() != self.gl.show_normals:
            self.norm_check.setChecked(self.gl.show_normals)
        if self.axes_check.isChecked() != self.gl.show_axes:
            self.axes_check.setChecked(self.gl.show_axes)

    def keyPressEvent(self, ev):
        """Обработка нажатия клавиш в главном окне"""
        if ev.key() == QtCore.Qt.Key_Space:
            self.gl.anim_running = not getattr(self.gl, 'anim_running', False)
            self.gl.update()
        else:
            super().keyPressEvent(ev)


def main():
    """Главная функция"""
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()