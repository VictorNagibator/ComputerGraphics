# lab_rev_opengl_fixed.py
# Полный рабочий файл — копируй и запускай.
# Требуется: PyQt5, PyOpenGL, numpy

import sys, math, time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL import GL
from OpenGL.GL import shaders

# ---------------------------
# GLSL шейдеры (Gouraud)
# ---------------------------
VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform vec3 uLightPosWorld;
uniform float uKa;
uniform float uKd;
uniform float uIa;
uniform float uIl;

out float vIntensity;

void main() {
    vec4 posWorld = vec4(aPos, 1.0);
    vec3 N = normalize(aNormal);
    vec3 L = normalize(uLightPosWorld - posWorld.xyz);
    float diff = max(0.0, dot(N, L));
    float I = uIa * uKa + uIl * uKd * diff;
    vIntensity = clamp(I, 0.0, 1.0);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

FRAG_SHADER = """
#version 330 core
in float vIntensity;
out vec4 FragColor;
void main() {
    float v = clamp(vIntensity, 0.0, 1.0);
    FragColor = vec4(v, v, v, 1.0);
}
"""

AXIS_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform vec3 uColor;
out vec3 vColor;
void main(){ vColor = uColor; gl_Position = uMVP * vec4(aPos,1.0); }
"""

AXIS_FRAG = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){ FragColor = vec4(vColor,1.0); }
"""

# ---------------------------
# Матричные утилиты
# ---------------------------
def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect
    M[1,1] = f
    M[2,2] = (zfar + znear) / (znear - zfar)
    M[2,3] = (2.0 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0
    return M

def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s
    M[1,0:3] = u
    M[2,0:3] = -f
    T = np.eye(4, dtype=np.float32)
    T[0:3,3] = -eye
    return M @ T

# ---------------------------
# Генераторы (raw triangulation)
# ---------------------------
def make_sphere_raw(radius=1.0, lat_segments=20, lon_segments=36):
    positions = []
    indices = []
    lat_segments = max(2, int(lat_segments))
    lon_segments = max(3, int(lon_segments))
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments
        y = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            positions.append((x,y,z))
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j
            b = (i + 1) * L + j
            c = (i + 1) * L + ((j + 1) % L)
            d = i * L + ((j + 1) % L)
            indices.extend((a,b,c)); indices.extend((a,c,d))
    return np.array(positions, dtype=np.float32), np.array(indices, dtype=np.uint32)

def make_ellipsoid_raw(rx=1.4, ry=0.8, rz=1.0, lat_segments=18, lon_segments=32):
    pos = []; idx = []
    lat_segments = max(2, int(lat_segments)); lon_segments = max(3, int(lon_segments))
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments
        cy = math.cos(phi); sy = math.sin(phi)
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            cx = math.cos(theta); sx = math.sin(theta)
            x = rx * sy * cx
            y = ry * cy
            z = rz * sy * sx
            pos.append((x,y,z))
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j
            b = (i + 1) * L + j
            c = (i + 1) * L + ((j + 1) % L)
            d = i * L + ((j + 1) % L)
            idx.extend((a,b,c)); idx.extend((a,c,d))
    return np.array(pos,dtype=np.float32), np.array(idx,dtype=np.uint32)

def make_torus_raw_with_normals(R=1.0, r=0.35, seg_major=36, seg_minor=18):
    pos=[]; norms=[]; idx=[]
    seg_major = max(3, int(seg_major)); seg_minor = max(3, int(seg_minor))
    for i in range(seg_major):
        u = 2.0*math.pi*i/seg_major
        cu = math.cos(u); su = math.sin(u)
        for j in range(seg_minor):
            v = 2.0*math.pi*j/seg_minor
            cv = math.cos(v); sv = math.sin(v)
            x = (R + r*cv) * cu
            y = r * sv
            z = (R + r*cv) * su
            pos.append((x,y,z))
            nx = cu * cv; ny = sv; nz = su * cv
            n = np.array((nx,ny,nz), dtype=np.float32)
            n /= (np.linalg.norm(n) + 1e-12)
            norms.append(tuple(n))
    M = seg_minor
    for i in range(seg_major):
        for j in range(seg_minor):
            a = i*M + j
            b = ((i+1)%seg_major)*M + j
            c = ((i+1)%seg_major)*M + ((j+1)%M)
            d = i*M + ((j+1)%M)
            idx.extend((a,b,c)); idx.extend((a,c,d))
    return np.array(pos,dtype=np.float32), np.array(norms,dtype=np.float32), np.array(idx,dtype=np.uint32)

# ---------------------------
# Нормали и ориентирование (ручное)
# ---------------------------
def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray):
    N = positions.shape[0]
    normals = np.zeros((N,3), dtype=np.float64)
    tri_count = indices.size // 3
    for t in range(tri_count):
        i0 = int(indices[3*t+0]); i1 = int(indices[3*t+1]); i2 = int(indices[3*t+2])
        p0 = positions[i0]; p1 = positions[i1]; p2 = positions[i2]
        v1 = p1 - p0; v2 = p2 - p0
        fn = np.cross(v1, v2)
        nlen = np.linalg.norm(fn)
        if nlen > 1e-12:
            fn = fn / nlen
        else:
            fn = np.array([0.0,0.0,0.0])
        normals[i0] += fn; normals[i1] += fn; normals[i2] += fn
    for i in range(N):
        l = np.linalg.norm(normals[i])
        if l < 1e-12:
            normals[i] = np.array([0.0,0.0,1.0])
        else:
            normals[i] /= l
    # force outward for roughly centered objects
    for i in range(N):
        if np.dot(normals[i], positions[i]) < 0.0:
            normals[i] = -normals[i]
    return normals.astype(np.float32)

def orient_triangles_outward(positions: np.ndarray, indices: np.ndarray):
    idx = indices.copy().astype(np.int64)
    for t in range(idx.size // 3):
        i0 = idx[3*t+0]; i1 = idx[3*t+1]; i2 = idx[3*t+2]
        p0 = positions[i0]; p1 = positions[i1]; p2 = positions[i2]
        fn = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(fn) < 1e-12:
            continue
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(fn, centroid) < 0.0:
            idx[3*t+1], idx[3*t+2] = idx[3*t+2], idx[3*t+1]
    return idx.astype(np.uint32)

# ---------------------------
# OpenGL Widget
# ---------------------------
class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat(); fmt.setDepthBufferSize(24); fmt.setProfile(QSurfaceFormat.CoreProfile); fmt.setVersion(3,3)
        self.setFormat(fmt)

        # модель/сетка
        self.shape = 'sphere'
        self.lat_segments = 24
        self.lon_segments = 40

        # камера (spherical)
        self.camera_target = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.cam_radius = 4.5
        self.cam_az = 0.0
        self.cam_el = math.radians(10.0)
        self.camera_up = np.array([0.0,1.0,0.0], dtype=np.float32)

        # свет
        self.light_world = np.array([2.5,3.0,2.5], dtype=np.float32)
        self.light_on = True
        self.Ka = 0.35; self.Kd = 1.0; self.Ia = 0.25; self.Il = 1.2

        # отображение
        self.show_wireframe = True
        self.show_normals = False
        self.show_axes = True

        # движение
        self.anim_running = False
        self.cam_move_active = False
        self.cam_move_start = 0.0
        self.cam_move_dur = 1500
        self.cam_move_phase_split = 0.25
        self.cam_move_from_eye = None
        self.cam_move_light_eye = None
        self.cam_move_to_eye = None

        # GL ids
        self.program = None; self.axis_program = None
        self.vao = None; self.vbo = None; self.ebo = None; self.index_count = 0
        self.axes_vao = None; self.axes_vbo = None
        self.light_vao = None; self.light_vbo = None
        self.normals_vao = None; self.normals_vbo = None; self.normals_count = 0

        # VAO support flag
        self.use_vao = False

        # ввод
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.mouse_mode = None; self.last_mouse = None

        # таймер
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.on_timer); self.timer.start(16)

    def initializeGL(self):
        # шейдеры
        try:
            vert = shaders.compileShader(VERT_SHADER, GL.GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAG_SHADER, GL.GL_FRAGMENT_SHADER)
            self.program = shaders.compileProgram(vert, frag)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка шейдера", str(e))
            self.program = None
            return

        try:
            vert2 = shaders.compileShader(AXIS_VERT, GL.GL_VERTEX_SHADER)
            frag2 = shaders.compileShader(AXIS_FRAG, GL.GL_FRAGMENT_SHADER)
            self.axis_program = shaders.compileProgram(vert2, frag2)
        except Exception as e:
            print("Axis shader failed:", e)
            self.axis_program = None

        # test VAO support (делаем в контексте initializeGL, где контекст гарантирован)
        try:
            vao_test = GL.glGenVertexArrays(1)
            # try to delete to be safe
            try:
                GL.glDeleteVertexArrays(1, [vao_test])
            except Exception:
                pass
            self.use_vao = True
        except Exception:
            print("VAO not available — fallback mode.")
            self.use_vao = False

        # создаём начальные меши/оси/маркер света
        self.create_mesh()
        self.create_axes()
        self.create_light_marker()
        self.create_normals_vbo()  # initial empty (filled inside create_mesh)

        # GL state
        GL.glEnable(GL.GL_DEPTH_TEST)
        try:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
            GL.glFrontFace(GL.GL_CCW)
        except Exception:
            pass
        GL.glClearColor(0.96,0.96,0.96,1.0)

    def create_mesh(self):
        # generate positions/indices depending on shape
        if self.shape == 'sphere':
            pos, idx = make_sphere_raw(radius=1.0, lat_segments=self.lat_segments, lon_segments=self.lon_segments)
        elif self.shape == 'ellipsoid':
            pos, idx = make_ellipsoid_raw(rx=1.4, ry=0.8, rz=1.0,
                                          lat_segments=max(3, self.lat_segments),
                                          lon_segments=max(3, self.lon_segments))
        elif self.shape == 'torus':
            pos, analytic_norms, idx = make_torus_raw_with_normals(R=1.0, r=0.4,
                                                                   seg_major=max(3,self.lon_segments),
                                                                   seg_minor=max(3, self.lat_segments//2))
        else:
            pos, idx = make_sphere_raw()

        # defensive: ensure triangulation outward and compute vertex normals per textbook
        idx = orient_triangles_outward(pos, idx)
        normals = compute_vertex_normals(pos, idx)

        # store arrays (keep for normals drawing)
        self._pos = pos
        self._normals = normals
        self._idx = idx

        # prepare interleaved buffer (pos + normal)
        verts_interleaved = np.ascontiguousarray(np.hstack((pos, normals)).astype(np.float32))

        self.index_count = idx.size

        # delete old buffers safely
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

        # create VBO and EBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_interleaved.nbytes, verts_interleaved, GL.GL_STATIC_DRAW)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

        # create/bind VAO if available
        if self.use_vao:
            try:
                self.vao = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(self.vao)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
                GL.glBindVertexArray(0)
            except Exception as e:
                print("VAO setup failed, disabling VAO:", e)
                self.use_vao = False
                try:
                    GL.glDeleteVertexArrays(1, [self.vao])
                except Exception:
                    pass
                self.vao = None

        # unbind buffers
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        # create normals VBO for display (lines per vertex)
        self.create_normals_vbo()

    def create_axes(self):
        axes = np.array([0,0,0, 1.5,0,0,  0,0,0, 0,1.5,0,  0,0,0, 0,0,1.5 ], dtype=np.float32)
        try:
            if getattr(self, 'axes_vao', None) and self.use_vao:
                GL.glDeleteVertexArrays(1, [self.axes_vao])
        except Exception:
            pass
        try:
            if getattr(self, 'axes_vbo', None):
                GL.glDeleteBuffers(1, [self.axes_vbo])
        except Exception:
            pass

        self.axes_vao = GL.glGenVertexArrays(1) if self.use_vao else None
        self.axes_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, axes.nbytes, axes, GL.GL_STATIC_DRAW)
        if self.use_vao:
            GL.glBindVertexArray(self.axes_vao)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def create_light_marker(self):
        try:
            if getattr(self, 'light_vao', None) and self.use_vao:
                GL.glDeleteVertexArrays(1, [self.light_vao])
        except Exception:
            pass
        try:
            if getattr(self, 'light_vbo', None):
                GL.glDeleteBuffers(1, [self.light_vbo])
        except Exception:
            pass

        self.light_vao = GL.glGenVertexArrays(1) if self.use_vao else None
        self.light_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 3*4, None, GL.GL_DYNAMIC_DRAW)
        if self.use_vao:
            GL.glBindVertexArray(self.light_vao)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def create_normals_vbo(self):
        # Build lines: for each vertex -> two points (pos, pos + normal * scale)
        if not hasattr(self, '_pos') or not hasattr(self, '_normals'):
            # nothing yet
            self.normals_count = 0
            return
        pos = self._pos
        normals = self._normals
        N = pos.shape[0]
        # scale relative to model size
        bbox = np.max(np.linalg.norm(pos, axis=1)) if N>0 else 1.0
        scale = 0.12 * (bbox if bbox>0 else 1.0)
        lines = np.zeros((N*2, 3), dtype=np.float32)
        for i in range(N):
            lines[2*i] = pos[i]
            lines[2*i+1] = pos[i] + normals[i] * scale
        self.normals_count = N*2

        # delete old
        try:
            if getattr(self, 'normals_vao', None) and self.use_vao:
                GL.glDeleteVertexArrays(1, [self.normals_vao])
        except Exception:
            pass
        try:
            if getattr(self, 'normals_vbo', None):
                GL.glDeleteBuffers(1, [self.normals_vbo])
        except Exception:
            pass

        self.normals_vao = GL.glGenVertexArrays(1) if self.use_vao else None
        self.normals_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, lines.nbytes, lines, GL.GL_STATIC_DRAW)
        if self.use_vao:
            GL.glBindVertexArray(self.normals_vao)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def resizeGL(self, w, h):
        GL.glViewport(0,0,w,h)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.program:
            return

        GL.glUseProgram(self.program)
        w = max(1,self.width()); h = max(1,self.height()); aspect = w/float(h)
        proj = perspective(math.radians(60.0), aspect, 0.1, 100.0)

        eye = self._sph_to_cart(self.cam_az, self.cam_el, self.cam_radius) + self.camera_target
        view = look_at(eye, self.camera_target, self.camera_up)
        mv = view @ np.eye(4, dtype=np.float32)
        mvp = proj @ mv

        # uniforms
        loc = GL.glGetUniformLocation(self.program, 'uMVP'); GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
        loc = GL.glGetUniformLocation(self.program, 'uLightPosWorld'); GL.glUniform3f(loc, float(self.light_world[0]), float(self.light_world[1]), float(self.light_world[2]))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKa'), float(self.Ka))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKd'), float(self.Kd))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIa'), float(self.Ia))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIl'), float(self.Il) if self.light_on else 0.0)

        # draw filled geometry
        if self.index_count > 0:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL); GL.glPolygonOffset(1.0, 1.0)
            if self.use_vao and self.vao is not None:
                GL.glBindVertexArray(self.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glDisableVertexAttribArray(0); GL.glDisableVertexAttribArray(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0); GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glUseProgram(0)

        # wireframe overlay
        if self.show_wireframe and self.axis_program:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP'); GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor'); GL.glUniform3f(locc, 0.08,0.08,0.08)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE); GL.glLineWidth(1.0)
            if self.use_vao and self.vao is not None:
                GL.glBindVertexArray(self.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0); GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glUseProgram(0)

        # normals lines (vertex normals)
        if self.show_normals and self.normals_count > 0 and self.axis_program:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP'); GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor'); GL.glUniform3f(locc, 0.0, 0.45, 0.8)
            if self.use_vao and self.normals_vao is not None:
                GL.glBindVertexArray(self.normals_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, int(self.normals_count))
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
                GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,GL.ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_LINES, 0, int(self.normals_count))
                GL.glDisableVertexAttribArray(0); GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

        # axes
        if self.show_axes and self.axis_program:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP'); GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            if self.use_vao and self.axes_vao is not None:
                GL.glUniform3f(locc, 1.0, 0.0, 0.0); GL.glBindVertexArray(self.axes_vao); GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glUniform3f(locc, 0.0, 1.0, 0.0); GL.glDrawArrays(GL.GL_LINES, 2, 2)
                GL.glUniform3f(locc, 0.0, 0.0, 1.0); GL.glDrawArrays(GL.GL_LINES, 4, 2)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
                GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,GL.ctypes.c_void_p(0))
                GL.glUniform3f(locc, 1.0, 0.0, 0.0); GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glUniform3f(locc, 0.0, 1.0, 0.0); GL.glDrawArrays(GL.GL_LINES, 2, 2)
                GL.glUniform3f(locc, 0.0, 0.0, 1.0); GL.glDrawArrays(GL.GL_LINES, 4, 2)
                GL.glDisableVertexAttribArray(0); GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

        # light marker (point)
        if self.light_vao is not None and self.axis_program:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP'); GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor'); GL.glUniform3f(locc, 1.0, 0.85, 0.0)
            # update light buffer
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
            data = np.array(self.light_world, dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
            if self.use_vao and self.light_vao is not None:
                GL.glBindVertexArray(self.light_vao)
                GL.glPointSize(10.0); GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glBindVertexArray(0)
            else:
                GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,0,GL.ctypes.c_void_p(0))
                GL.glPointSize(10.0); GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glDisableVertexAttribArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

    def _sph_to_cart(self, az, el, r):
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        return np.array([x,y,z], dtype=np.float32)

    # keyboard: WASD pan target in world-space
    def keyPressEvent(self, ev):
        k = ev.key(); step = 0.15
        eye = self._sph_to_cart(self.cam_az, self.cam_el, self.cam_radius) + self.camera_target
        forward = self.camera_target - eye; forward /= (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, self.camera_up); right /= (np.linalg.norm(right) + 1e-12)
        up = self.camera_up
        if k == QtCore.Qt.Key_W:
            self.camera_target = self.camera_target + up * step; self.update()
        elif k == QtCore.Qt.Key_S:
            self.camera_target = self.camera_target - up * step; self.update()
        elif k == QtCore.Qt.Key_A:
            self.camera_target = self.camera_target - right * step; self.update()
        elif k == QtCore.Qt.Key_D:
            self.camera_target = self.camera_target + right * step; self.update()
        elif k == QtCore.Qt.Key_Space:
            self.anim_running = not self.anim_running; self.update()
        else:
            super().keyPressEvent(ev)

    # mouse orbit
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouse_mode = 'orbit'; self._orbit_start = (self.cam_az, self.cam_el, self.cam_radius); self._orbit_mouse = ev.pos()
        self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_mouse is None: return
        if self.mouse_mode == 'orbit':
            az0, el0, r0 = self._orbit_start
            ddx = ev.x() - self._orbit_mouse.x(); ddy = ev.y() - self._orbit_mouse.y()
            az = az0 + ddx * 0.01; el = el0 + ddy * 0.01
            max_el = math.radians(89.0); el = max(-max_el, min(max_el, el))
            self.cam_az = az; self.cam_el = el; self.update()
        self.last_mouse = ev.pos()

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None; self.last_mouse = None

    # start movement: плавно current -> light -> -light along spherical arc
    def start_camera_move_via_light(self, duration_ms=1500):
        eye_current = self._sph_to_cart(self.cam_az, self.cam_el, self.cam_radius) + self.camera_target
        eye_light = np.array(self.light_world, dtype=np.float32)
        eye_opposite = -eye_light
        self.cam_move_from_eye = eye_current
        self.cam_move_light_eye = eye_light
        self.cam_move_to_eye = eye_opposite
        self.cam_move_start = time.time()*1000.0
        self.cam_move_dur = max(50, int(duration_ms))
        self.cam_move_active = True

    # slerp helper for directions around camera_target
    def _slerp_dir(self, v0, v1, t):
        n0 = v0 / (np.linalg.norm(v0) + 1e-12)
        n1 = v1 / (np.linalg.norm(v1) + 1e-12)
        dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
        ang = math.acos(dot)
        if abs(ang) < 1e-6:
            out = (1.0 - t) * n0 + t * n1
            out /= (np.linalg.norm(out) + 1e-12)
            return out
        out = (math.sin((1.0 - t) * ang) * n0 + math.sin(t * ang) * n1) / (math.sin(ang) + 1e-12)
        out /= (np.linalg.norm(out) + 1e-12)
        return out

    def on_timer(self):
        now = time.time()*1000.0
        if self.anim_running:
            self.cam_az += 0.01; self.update()
        if self.cam_move_active:
            t_full = (now - self.cam_move_start) / float(self.cam_move_dur)
            if t_full >= 1.0:
                t_full = 1.0; self.cam_move_active = False
            p = self.cam_move_phase_split
            if t_full <= p:
                s = t_full / (p + 1e-12)
                eye0 = self.cam_move_from_eye; eye1 = self.cam_move_light_eye
                v0 = eye0 - self.camera_target; v1 = eye1 - self.camera_target
                dirn = self._slerp_dir(v0, v1, s)
                r = (1.0 - s) * np.linalg.norm(v0) + s * np.linalg.norm(v1)
                eye = self.camera_target + dirn * r
            else:
                s = (t_full - p) / (1.0 - p + 1e-12)
                v_light = self.cam_move_light_eye - self.camera_target
                v_op = self.cam_move_to_eye - self.camera_target
                r_light = np.linalg.norm(v_light)
                dirn = self._slerp_dir(v_light, v_op, s)
                eye = self.camera_target + dirn * r_light
            v = eye - self.camera_target
            r = np.linalg.norm(v)
            if r < 1e-6:
                r = 1e-6; v = np.array([r,0,0], dtype=np.float32)
            el = math.asin(v[1]/r)
            az = math.atan2(v[2], v[0])
            self.cam_radius = float(r); self.cam_el = float(el); self.cam_az = float(az)
            self.update()

# ---------------------------
# GUI (русский)
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Лабораторная: Шар + OpenGL (Gouraud)')
        self.gl = GLWidget(); self.setCentralWidget(self.gl)
        self.create_controls(); self.resize(1300,820)
        self.ui_timer = QtCore.QTimer(); self.ui_timer.timeout.connect(self.update_ui); self.ui_timer.start(100)

    def create_controls(self):
        panel = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel('<b>Управление сценой</b>'))

        # формы
        shape_h = QtWidgets.QHBoxLayout()
        btn_sphere = QtWidgets.QPushButton('Сфера'); btn_sphere.clicked.connect(lambda: self.set_shape('sphere'))
        btn_ell = QtWidgets.QPushButton('Эллипсоид'); btn_ell.clicked.connect(lambda: self.set_shape('ellipsoid'))
        btn_tor = QtWidgets.QPushButton('Тор'); btn_tor.clicked.connect(lambda: self.set_shape('torus'))
        shape_h.addWidget(btn_sphere); shape_h.addWidget(btn_ell); shape_h.addWidget(btn_tor)
        layout.addLayout(shape_h)

        # сетка (в live режиме)
        layout.addWidget(QtWidgets.QLabel('Настройки сетки (lat x lon)'))
        grid_h = QtWidgets.QHBoxLayout()
        self.lat_spin = QtWidgets.QSpinBox(); self.lat_spin.setRange(3,160); self.lat_spin.setValue(self.gl.lat_segments)
        self.lon_spin = QtWidgets.QSpinBox(); self.lon_spin.setRange(3,320); self.lon_spin.setValue(self.gl.lon_segments)
        self.lat_spin.valueChanged.connect(self.on_mesh_changed); self.lon_spin.valueChanged.connect(self.on_mesh_changed)
        grid_h.addWidget(self.lat_spin); grid_h.addWidget(self.lon_spin)
        layout.addLayout(grid_h)

        # отображения
        self.tri_check = QtWidgets.QCheckBox('Показывать триангуляцию (wireframe)'); self.tri_check.setChecked(self.gl.show_wireframe)
        self.tri_check.stateChanged.connect(lambda s: self.on_wire_toggle(s))
        self.norm_check = QtWidgets.QCheckBox('Показывать векторы нормалей (по вершинам)'); self.norm_check.setChecked(self.gl.show_normals)
        self.norm_check.stateChanged.connect(lambda s: self.on_normals_toggle(s))
        self.axes_check = QtWidgets.QCheckBox('Показывать оси'); self.axes_check.setChecked(self.gl.show_axes)
        self.axes_check.stateChanged.connect(lambda s: self.on_axes_toggle(s))
        layout.addWidget(self.tri_check); layout.addWidget(self.norm_check); layout.addWidget(self.axes_check)

        # свет
        layout.addWidget(QtWidgets.QLabel('<b>Свет</b>'))
        self.light_check = QtWidgets.QCheckBox('Включить свет'); self.light_check.setChecked(True); self.light_check.stateChanged.connect(self.on_light_toggle)
        layout.addWidget(self.light_check)
        coords_h = QtWidgets.QHBoxLayout()
        self.lx = QtWidgets.QDoubleSpinBox(); self.lx.setRange(-20.0,20.0); self.lx.setSingleStep(0.1); self.lx.setValue(float(self.gl.light_world[0])); self.lx.valueChanged.connect(self.on_light_changed)
        self.ly = QtWidgets.QDoubleSpinBox(); self.ly.setRange(-20.0,20.0); self.ly.setSingleStep(0.1); self.ly.setValue(float(self.gl.light_world[1])); self.ly.valueChanged.connect(self.on_light_changed)
        self.lz = QtWidgets.QDoubleSpinBox(); self.lz.setRange(-20.0,20.0); self.lz.setSingleStep(0.1); self.lz.setValue(float(self.gl.light_world[2])); self.lz.valueChanged.connect(self.on_light_changed)
        coords_h.addWidget(QtWidgets.QLabel('x')); coords_h.addWidget(self.lx); coords_h.addWidget(QtWidgets.QLabel('y')); coords_h.addWidget(self.ly); coords_h.addWidget(QtWidgets.QLabel('z')); coords_h.addWidget(self.lz)
        layout.addLayout(coords_h)

        # параметры освещения
        params_h = QtWidgets.QGridLayout()
        params_h.addWidget(QtWidgets.QLabel('Ka (ambient)'), 0, 0)
        self.ka_spin = QtWidgets.QDoubleSpinBox(); self.ka_spin.setRange(0.0,2.0); self.ka_spin.setSingleStep(0.05); self.ka_spin.setValue(self.gl.Ka); self.ka_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ka_spin, 0, 1)
        params_h.addWidget(QtWidgets.QLabel('Kd (diffuse)'), 1, 0)
        self.kd_spin = QtWidgets.QDoubleSpinBox(); self.kd_spin.setRange(0.0,2.0); self.kd_spin.setSingleStep(0.05); self.kd_spin.setValue(self.gl.Kd); self.kd_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.kd_spin, 1, 1)
        params_h.addWidget(QtWidgets.QLabel('Ia (ambient intensity)'), 2, 0)
        self.ia_spin = QtWidgets.QDoubleSpinBox(); self.ia_spin.setRange(0.0,2.0); self.ia_spin.setSingleStep(0.05); self.ia_spin.setValue(self.gl.Ia); self.ia_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ia_spin, 2, 1)
        params_h.addWidget(QtWidgets.QLabel('Il (light intensity)'), 3, 0)
        self.il_spin = QtWidgets.QDoubleSpinBox(); self.il_spin.setRange(0.0,5.0); self.il_spin.setSingleStep(0.05); self.il_spin.setValue(self.gl.Il); self.il_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.il_spin, 3, 1)
        layout.addLayout(params_h)

        # движение (свет -> противоположная точка по дуге)
        move_h = QtWidgets.QHBoxLayout()
        self.btn_move = QtWidgets.QPushButton('Движение: свет → противоположная (дуга)')
        self.btn_move.clicked.connect(self.on_start_move)
        move_h.addWidget(self.btn_move)
        layout.addLayout(move_h)

        sp_h = QtWidgets.QHBoxLayout()
        sp_h.addWidget(QtWidgets.QLabel('Скорость движения (ms)'))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.speed_slider.setRange(200,5000); self.speed_slider.setValue(1500)
        self.speed_label = QtWidgets.QLabel('1500'); self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        sp_h.addWidget(self.speed_slider); sp_h.addWidget(self.speed_label)
        layout.addLayout(sp_h)

        # авто/стоп
        h = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton('Авто'); self.btn_play.setCheckable(True); self.btn_play.clicked.connect(self.on_play)
        btn_stop = QtWidgets.QPushButton('Остановить'); btn_stop.clicked.connect(self.on_stop)
        h.addWidget(self.btn_play); h.addWidget(btn_stop); layout.addLayout(h)

        instr = QtWidgets.QLabel('W/S/A/D — пан по XYZ; Space — авто-вращение; ЛКМ+перетаскивание — орбита')
        instr.setWordWrap(True); layout.addWidget(instr)

        # метки координат
        layout.addWidget(QtWidgets.QLabel('<b>Камера (eye)</b>'))
        self.camera_label = QtWidgets.QLabel('x: 0.00, y: 0.00, z: 0.00'); layout.addWidget(self.camera_label)
        layout.addWidget(QtWidgets.QLabel('<b>Источник света (координаты)</b>'))
        self.light_label = QtWidgets.QLabel('x: 0.00, y: 0.00, z: 0.00'); layout.addWidget(self.light_label)

        layout.addStretch()
        panel.setLayout(layout)
        dock = QtWidgets.QDockWidget('Панель управления', self); dock.setWidget(panel); self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # UI callbacks
    def set_shape(self, s):
        self.gl.makeCurrent()
        self.gl.shape = s
        self.gl.create_mesh()
        self.gl.doneCurrent()
        self.gl.update()

    def on_mesh_changed(self, val=None):
        try:
            self.gl.makeCurrent()
            self.gl.lat_segments = int(self.lat_spin.value())
            self.gl.lon_segments = int(self.lon_spin.value())
            self.gl.create_mesh()
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    def on_wire_toggle(self, state):
        self.gl.show_wireframe = (state == QtCore.Qt.Checked); self.gl.update()

    def on_normals_toggle(self, state):
        self.gl.show_normals = (state == QtCore.Qt.Checked); self.gl.update()

    def on_axes_toggle(self, state):
        self.gl.show_axes = (state == QtCore.Qt.Checked); self.gl.update()

    def on_light_toggle(self, state):
        self.gl.light_on = (state == QtCore.Qt.Checked); self.gl.update()

    def on_light_changed(self, val=None):
        try:
            self.gl.makeCurrent()
            self.gl.light_world = np.array([float(self.lx.value()), float(self.ly.value()), float(self.lz.value())], dtype=np.float32)
            self.gl.create_light_marker()
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    def on_light_params_changed(self, val=None):
        self.gl.Ka = float(self.ka_spin.value()); self.gl.Kd = float(self.kd_spin.value())
        self.gl.Ia = float(self.ia_spin.value()); self.gl.Il = float(self.il_spin.value())
        self.gl.update()

    def on_start_move(self):
        dur = int(self.speed_slider.value())
        self.gl.start_camera_move_via_light(duration_ms=dur)
        self.gl.update()

    def on_play(self, checked):
        if checked:
            self.btn_play.setText('Вращается...'); self.gl.anim_running = True
        else:
            self.btn_play.setText('Авто'); self.gl.anim_running = False
        self.gl.update()

    def on_stop(self):
        self.btn_play.setChecked(False); self.btn_play.setText('Авто'); self.gl.anim_running = False; self.gl.update()

    def update_ui(self):
        lw = self.gl.light_world
        self.light_label.setText(f'x: {lw[0]:.2f}, y: {lw[1]:.2f}, z: {lw[2]:.2f}')
        eye = self.gl._sph_to_cart(self.gl.cam_az, self.gl.cam_el, self.gl.cam_radius) + self.gl.camera_target
        self.camera_label.setText(f'x: {eye[0]:.2f}, y: {eye[1]:.2f}, z: {eye[2]:.2f}')
        # keep checkboxes consistent if changed externally
        if self.tri_check.isChecked() != self.gl.show_wireframe:
            self.tri_check.setChecked(self.gl.show_wireframe)
        if self.norm_check.isChecked() != self.gl.show_normals:
            self.norm_check.setChecked(self.gl.show_normals)
        if self.axes_check.isChecked() != self.gl.show_axes:
            self.axes_check.setChecked(self.gl.show_axes)

    def keyPressEvent(self, ev):
        self.gl.keyPressEvent(ev)

# ---------------------------
# Запуск
# ---------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
