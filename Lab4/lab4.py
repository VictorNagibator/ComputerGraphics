"""
Bikini Bottom — статическая 3D-сцена на Python + PyOpenGL + GLFW

Обновлённая версия — правки по последним замечаниям пользователя:
1) небо разделено: сверху sky.png, по бокам background.png
2) рамка двери Спанчбоба повторяет форму полуэллипса

Файлы текстур (положить рядом в папке `textures/`):
    textures/sand.png
    textures/sky.png
    textures/background.png
    textures/pineapple.png
    textures/rock.png
    textures/squidward.png
    textures/leaf.png
    textures/road.png
    textures/wood.png
    textures/metal.png

Зависимости:
    pip install PyOpenGL PyOpenGL_accelerate glfw Pillow PyGLM numpy

Запуск:
    python bikini_bottom_python.py
"""

from OpenGL.GL import *
import glfw
import numpy as np
from PIL import Image
import ctypes
from pyglm import glm
import os
import sys

# ---------- Шейдеры (GLSL)
VERT_SHADER = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTex;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 FragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main(){
    FragPos = vec3(model * vec4(aPos,1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTex;
    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos,1.0);
    gl_Position = projection * view * vec4(FragPos,1.0);
}
"""

FRAG_SHADER = """#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform sampler2D texture_diffuse1;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float materialShininess;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir){
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if(projCoords.z > 1.0) return 0.0;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x=-1;x<=1;x++){
        for(int y=-1;y<=1;y++){
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y)*texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}

void main(){
    vec3 color = texture(texture_diffuse1, TexCoord).rgb;
    vec3 normal = normalize(Normal);
    vec3 lightColor = vec3(1.0);
    vec3 ambient = 0.2 * color;
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
    vec3 specular = spec * lightColor * 0.5;
    float shadow = ShadowCalculation(FragPosLightSpace, normal, lightDir);
    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(lighting, 1.0);
}
"""

DEPTH_VS = """#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main(){
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
"""

DEPTH_FS = """#version 330 core
void main(){}
"""

# ---------- Помощники для шейдеров

def compile_shader(src, type):
    shader = glCreateShader(type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    success = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not success:
        info = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error: {info}")
    return shader


def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    success = glGetProgramiv(prog, GL_LINK_STATUS)
    if not success:
        info = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error: {info}")
    glDeleteShader(vs); glDeleteShader(fs)
    return prog

# ---------- Примитивы (плоскость, куб, цилиндр, сфера, диск)

def create_plane(size=50.0, uv_scale=1.0):
    s = size
    verts = []
    coords = [(-s,0.0,-s),( s,0.0, s),( s,0.0,-s),( s,0.0, s),(-s,0.0,-s),(-s,0.0, s)]
    for x,y,z in coords:
        u = (x) / uv_scale
        v = (z) / uv_scale
        verts.extend([x,y,z, 0.0,1.0,0.0, u, v])
    return np.array(verts, dtype=np.float32)


def create_cube():
    verts = []
    def pushFace(a,b,c,d,n):
        verts.extend([*a,*n,0.0,0.0])
        verts.extend([*b,*n,1.0,1.0])
        verts.extend([*c,*n,1.0,0.0])
        verts.extend([*b,*n,1.0,1.0])
        verts.extend([*a,*n,0.0,0.0])
        verts.extend([*d,*n,0.0,1.0])
    pushFace((-0.5,-0.5,-0.5),(0.5,0.5,-0.5),(0.5,-0.5,-0.5),(-0.5,0.5,-0.5),(0,0,-1))
    pushFace((-0.5,-0.5,0.5),(0.5,0.5,0.5),(0.5,-0.5,0.5),(-0.5,0.5,0.5),(0,0,1))
    pushFace((-0.5,-0.5,-0.5),(-0.5,0.5,0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,0.5),(-1,0,0))
    pushFace((0.5,-0.5,-0.5),(0.5,0.5,0.5),(0.5,0.5,-0.5),(0.5,-0.5,0.5),(1,0,0))
    pushFace((-0.5,-0.5,-0.5),(0.5,-0.5,0.5),(0.5,-0.5,-0.5),(-0.5,-0.5,0.5),(0,-1,0))
    pushFace((-0.5,0.5,-0.5),(0.5,0.5,0.5),(0.5,0.5,-0.5),(-0.5,0.5,0.5),(0,1,0))
    return np.array(verts, dtype=np.float32)


def create_cylinder(segments=64):
    verts = []
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        p0 = (0.5*np.cos(a0), -0.5, 0.5*np.sin(a0))
        p1 = (0.5*np.cos(a1), -0.5, 0.5*np.sin(a1))
        p2 = (0.5*np.cos(a0),  0.5, 0.5*np.sin(a0))
        p3 = (0.5*np.cos(a1),  0.5, 0.5*np.sin(a1))
        n0 = (np.cos(a0),0,np.sin(a0))
        n1 = (np.cos(a1),0,np.sin(a1))
        verts.extend([*p0,*n0, i/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p3,*n1, (i+1)/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
    return np.array(verts, dtype=np.float32)


def create_sphere(lat=32, lon=32):
    verts = []
    for i in range(lat):
        theta1 = np.pi * i / lat
        theta2 = np.pi * (i+1) / lat
        for j in range(lon):
            phi1 = 2*np.pi * j / lon
            phi2 = 2*np.pi * (j+1) / lon
            p1 = (np.sin(theta1)*np.cos(phi1), np.cos(theta1), np.sin(theta1)*np.sin(phi1))
            p2 = (np.sin(theta2)*np.cos(phi1), np.cos(theta2), np.sin(theta2)*np.sin(phi1))
            p3 = (np.sin(theta1)*np.cos(phi2), np.cos(theta1), np.sin(theta1)*np.sin(phi2))
            p4 = (np.sin(theta2)*np.cos(phi2), np.cos(theta2), np.sin(theta2)*np.sin(phi2))
            verts.extend([*p1,*p1, j/lon, i/lat])
            verts.extend([*p2,*p2, j/lon, (i+1)/lat])
            verts.extend([*p3,*p3, (j+1)/lon, i/lat])
            verts.extend([*p3,*p3, (j+1)/lon, i/lat])
            verts.extend([*p2,*p2, j/lon, (i+1)/lat])
            verts.extend([*p4,*p4, (j+1)/lon, (i+1)/lat])
    return np.array(verts, dtype=np.float32)


def create_disk(segments=64):
    verts = []
    center = (0.0, 0.0, 0.0)
    normal = (0.0, 1.0, 0.0)
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        p0 = (0.5*np.cos(a0), 0.0, 0.5*np.sin(a0))
        p1 = (0.5*np.cos(a1), 0.0, 0.5*np.sin(a1))
        verts.extend([*center, *normal, 0.5, 0.5])
        verts.extend([*p0, *normal, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p1, *normal, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
    return np.array(verts, dtype=np.float32)


def create_window_frame(outer_radius=0.5, inner_radius=0.35, thickness=0.05, segments=32):
    """Создает объемную рамку окна (тороид)"""
    verts = []
    
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        
        # Внешние точки (с учетом толщины)
        p0_outer_front = (outer_radius*np.cos(a0), outer_radius*np.sin(a0), thickness/2)
        p1_outer_front = (outer_radius*np.cos(a1), outer_radius*np.sin(a1), thickness/2)
        p0_outer_back = (outer_radius*np.cos(a0), outer_radius*np.sin(a0), -thickness/2)
        p1_outer_back = (outer_radius*np.cos(a1), outer_radius*np.sin(a1), -thickness/2)
        
        # Внутренние точки (с учетом толщины)
        p0_inner_front = (inner_radius*np.cos(a0), inner_radius*np.sin(a0), thickness/2)
        p1_inner_front = (inner_radius*np.cos(a1), inner_radius*np.sin(a1), thickness/2)
        p0_inner_back = (inner_radius*np.cos(a0), inner_radius*np.sin(a0), -thickness/2)
        p1_inner_back = (inner_radius*np.cos(a1), inner_radius*np.sin(a1), -thickness/2)
        
        # Нормали
        n_outer = (np.cos(a0), np.sin(a0), 0)
        n_inner = (-np.cos(a0), -np.sin(a0), 0)
        n_top = (0, 0, 1)
        n_bottom = (0, 0, -1)
        
        # Внешняя боковая поверхность
        verts.extend([*p0_outer_front, *n_outer, i/segments, 0])
        verts.extend([*p0_outer_back, *n_outer, i/segments, 1])
        verts.extend([*p1_outer_front, *n_outer, (i+1)/segments, 0])
        verts.extend([*p1_outer_front, *n_outer, (i+1)/segments, 0])
        verts.extend([*p0_outer_back, *n_outer, i/segments, 1])
        verts.extend([*p1_outer_back, *n_outer, (i+1)/segments, 1])
        
        # Внутренняя боковая поверхность
        verts.extend([*p0_inner_front, *n_inner, i/segments, 0])
        verts.extend([*p1_inner_front, *n_inner, (i+1)/segments, 0])
        verts.extend([*p0_inner_back, *n_inner, i/segments, 1])
        verts.extend([*p1_inner_front, *n_inner, (i+1)/segments, 0])
        verts.extend([*p1_inner_back, *n_inner, (i+1)/segments, 1])
        verts.extend([*p0_inner_back, *n_inner, i/segments, 1])
        
        # Верхняя поверхность (передняя)
        verts.extend([*p0_outer_front, *n_top, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p1_outer_front, *n_top, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p0_inner_front, *n_top, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_outer_front, *n_top, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p1_inner_front, *n_top, 0.5+0.45*np.cos(a1), 0.5+0.45*np.sin(a1)])
        verts.extend([*p0_inner_front, *n_top, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        
        # Нижняя поверхность (задняя)
        verts.extend([*p0_outer_back, *n_bottom, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p0_inner_back, *n_bottom, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_outer_back, *n_bottom, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p1_outer_back, *n_bottom, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p0_inner_back, *n_bottom, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_inner_back, *n_bottom, 0.5+0.45*np.cos(a1), 0.5+0.45*np.sin(a1)])
    
    return np.array(verts, dtype=np.float32)


def create_skybox_sides():
    """Создает боковые стороны неба (4 стены)"""
    verts = []
    size = 100.0
    
    # Передняя грань
    verts.extend([-size, -size,  size,  0,0,-1,  0,0])
    verts.extend([ size, -size,  size,  0,0,-1,  1,0])
    verts.extend([ size,  size,  size,  0,0,-1,  1,1])
    verts.extend([ size,  size,  size,  0,0,-1,  1,1])
    verts.extend([-size,  size,  size,  0,0,-1,  0,1])
    verts.extend([-size, -size,  size,  0,0,-1,  0,0])
    
    # Задняя грань
    verts.extend([-size, -size, -size,  0,0,1,  1,0])
    verts.extend([ size,  size, -size,  0,0,1,  0,1])
    verts.extend([ size, -size, -size,  0,0,1,  0,0])
    verts.extend([ size,  size, -size,  0,0,1,  0,1])
    verts.extend([-size, -size, -size,  0,0,1,  1,0])
    verts.extend([-size,  size, -size,  0,0,1,  1,1])
    
    # Левая грань
    verts.extend([-size,  size,  size,  1,0,0,  1,0])
    verts.extend([-size,  size, -size,  1,0,0,  1,1])
    verts.extend([-size, -size, -size,  1,0,0,  0,1])
    verts.extend([-size, -size, -size,  1,0,0,  0,1])
    verts.extend([-size, -size,  size,  1,0,0,  0,0])
    verts.extend([-size,  size,  size,  1,0,0,  1,0])
    
    # Правая грань
    verts.extend([ size,  size,  size, -1,0,0,  0,0])
    verts.extend([ size, -size, -size, -1,0,0,  1,1])
    verts.extend([ size,  size, -size, -1,0,0,  1,0])
    verts.extend([ size, -size, -size, -1,0,0,  1,1])
    verts.extend([ size,  size,  size, -1,0,0,  0,0])
    verts.extend([ size, -size,  size, -1,0,0,  0,1])
    
    return np.array(verts, dtype=np.float32)


def create_skybox_top():
    """Создает верхнюю часть неба"""
    verts = []
    size = 100.0
    
    # Верхняя грань
    verts.extend([-size,  size, -size,  0,-1,0,  0,1])
    verts.extend([ size,  size,  size,  0,-1,0,  1,0])
    verts.extend([ size,  size, -size,  0,-1,0,  1,1])
    verts.extend([ size,  size,  size,  0,-1,0,  1,0])
    verts.extend([-size,  size, -size,  0,-1,0,  0,1])
    verts.extend([-size,  size,  size,  0,-1,0,  0,0])
    
    return np.array(verts, dtype=np.float32)


def create_door_frame(width=0.4, height=1.3, thickness=0.05, segments=16):
    """Создает изогнутую рамку для двери в форме полуэллипса"""
    verts = []
    
    # Полуэллипс для рамки двери
    for i in range(segments):
        # Углы для полуэллипса (от -pi/2 до pi/2)
        a0 = -np.pi/2 + np.pi * i / segments
        a1 = -np.pi/2 + np.pi * (i+1) / segments
        
        # Внешние точки эллипса
        x0_outer = width/2 * np.cos(a0)
        y0_outer = height/2 * np.sin(a0) + height/2
        x1_outer = width/2 * np.cos(a1)
        y1_outer = height/2 * np.sin(a1) + height/2
        
        # Внутренние точки эллипса
        x0_inner = (width/2 - thickness) * np.cos(a0)
        y0_inner = (height/2 - thickness) * np.sin(a0) + height/2
        x1_inner = (width/2 - thickness) * np.cos(a1)
        y1_inner = (height/2 - thickness) * np.sin(a1) + height/2
        
        # Нормали (направлены наружу от эллипса)
        n0 = (np.cos(a0), np.sin(a0), 0)
        n1 = (np.cos(a1), np.sin(a1), 0)
        
        # Передняя поверхность рамки
        verts.extend([x0_outer, y0_outer, thickness/2, *n0, i/segments, 0])
        verts.extend([x1_outer, y1_outer, thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x0_inner, y0_inner, thickness/2, *n0, i/segments, 1])
        
        verts.extend([x1_outer, y1_outer, thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x1_inner, y1_inner, thickness/2, *n1, (i+1)/segments, 1])
        verts.extend([x0_inner, y0_inner, thickness/2, *n0, i/segments, 1])
        
        # Задняя поверхность рамки
        verts.extend([x0_outer, y0_outer, -thickness/2, *n0, i/segments, 0])
        verts.extend([x0_inner, y0_inner, -thickness/2, *n0, i/segments, 1])
        verts.extend([x1_outer, y1_outer, -thickness/2, *n1, (i+1)/segments, 0])
        
        verts.extend([x1_outer, y1_outer, -thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x0_inner, y0_inner, -thickness/2, *n0, i/segments, 1])
        verts.extend([x1_inner, y1_inner, -thickness/2, *n1, (i+1)/segments, 1])
        
        # Боковая поверхность (внешняя)
        verts.extend([x0_outer, y0_outer, thickness/2, *n0, i/segments, 0])
        verts.extend([x0_outer, y0_outer, -thickness/2, *n0, i/segments, 1])
        verts.extend([x1_outer, y1_outer, thickness/2, *n1, (i+1)/segments, 0])
        
        verts.extend([x1_outer, y1_outer, thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x0_outer, y0_outer, -thickness/2, *n0, i/segments, 1])
        verts.extend([x1_outer, y1_outer, -thickness/2, *n1, (i+1)/segments, 1])
        
        # Боковая поверхность (внутренняя)
        verts.extend([x0_inner, y0_inner, thickness/2, *n0, i/segments, 0])
        verts.extend([x1_inner, y1_inner, thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x0_inner, y0_inner, -thickness/2, *n0, i/segments, 1])
        
        verts.extend([x1_inner, y1_inner, thickness/2, *n1, (i+1)/segments, 0])
        verts.extend([x1_inner, y1_inner, -thickness/2, *n1, (i+1)/segments, 1])
        verts.extend([x0_inner, y0_inner, -thickness/2, *n0, i/segments, 1])
    
    return np.array(verts, dtype=np.float32)

# ---------- Загрузка текстуры

def load_texture(path):
    img = Image.open(path).convert('RGBA')
    img_data = np.array(img)[::-1]  # flip vertically
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    try:
        max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, min(8.0, max_aniso))
    except Exception:
        pass
    return tex

# ---------- GL буферы/VAO

def make_vao(data, stride_elems=8):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    stride = stride_elems * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(6*ctypes.sizeof(ctypes.c_float)))
    glBindVertexArray(0)
    return vao, vbo, int(len(data)//stride_elems)

# ---------- Утилиты рисования параллелепипеда (масштабированный куб)

def model_for_box(center, size):
    return glm.translate(glm.mat4(1.0), glm.vec3(*center)) * glm.scale(glm.mat4(1.0), glm.vec3(*size))

# ---------- Main

def main():
    if not glfw.init():
        print('GLFW init failed')
        return
    width, height = 1280, 720
    window = glfw.create_window(width, height, 'Lab4 - Bikini Bottom', None, None)
    if not window:
        print('Window creation failed')
        glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # захватываем курсор для FPS-камеры
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    prog = link_program(VERT_SHADER, FRAG_SHADER)
    depth_prog = link_program(DEPTH_VS, DEPTH_FS)

    # создать меши
    plane_data = create_plane(40.0, uv_scale=30.0)
    plane_vao, _, plane_count = make_vao(plane_data)
    cube_data = create_cube()
    cube_vao, _, cube_count = make_vao(cube_data)
    cyl_data = create_cylinder(96)
    cyl_vao, _, cyl_count = make_vao(cyl_data)
    sph_data = create_sphere(48,48)
    sph_vao, _, sph_count = make_vao(sph_data)
    disk_data = create_disk(64)
    disk_vao, _, disk_count = make_vao(disk_data)
    window_frame_data = create_window_frame(thickness=0.08)
    window_frame_vao, _, window_frame_count = make_vao(window_frame_data)
    skybox_sides_data = create_skybox_sides()
    skybox_sides_vao, _, skybox_sides_count = make_vao(skybox_sides_data)
    skybox_top_data = create_skybox_top()
    skybox_top_vao, _, skybox_top_count = make_vao(skybox_top_data)
    door_frame_data = create_door_frame()
    door_frame_vao, _, door_frame_count = make_vao(door_frame_data)

    # текстуры — ищем в папке textures
    texdir = os.path.join(os.path.dirname(__file__), 'textures')
    tex_sand = load_texture(os.path.join(texdir, 'sand.png'))
    tex_sky = load_texture(os.path.join(texdir, 'sky.png'))
    tex_background = load_texture(os.path.join(texdir, 'background.png'))
    tex_pine = load_texture(os.path.join(texdir, 'pineapple.png'))
    tex_rock = load_texture(os.path.join(texdir, 'rock.png'))
    tex_squid = load_texture(os.path.join(texdir, 'squidward.png'))
    tex_leaf = load_texture(os.path.join(texdir, 'leaf.png'))
    tex_road = load_texture(os.path.join(texdir, 'road.png'))
    tex_wood = load_texture(os.path.join(texdir, 'wood.png'))
    tex_metal = load_texture(os.path.join(texdir, 'metal.png'))
    
    # Создаем простые однотонные текстуры для окон
    def create_color_texture(r, g, b, a=1.0):
        data = np.array([[int(r*255), int(g*255), int(b*255), int(a*255)]], dtype=np.uint8)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex
    
    tex_window_blue = create_color_texture(0.5, 0.7, 1.0)  # Голубое стекло
    tex_window_frame = create_color_texture(0.2, 0.3, 0.8)  # Синяя рамка

    # теневой фреймбуфер
    SHADOW_W, SHADOW_H = 2048, 2048
    depth_map_fbo = glGenFramebuffers(1)
    depth_map = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_map)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_W, SHADOW_H, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    border = (GLfloat * 4)(1.0,1.0,1.0,1.0)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border)
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0)
    glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print('Depth framebuffer not complete')
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    glEnable(GL_DEPTH_TEST)

    # камера - теперь FPS-подобная
    yaw, pitch = -90.0, 0.0
    cam_pos = glm.vec3(0.0, 3.0, 18.0)
    cam_front = glm.vec3(0.0, 0.0, -1.0)
    cam_up = glm.vec3(0.0, 1.0, 0.0)
    lastX, lastY = width/2, height/2
    first_mouse = True
    movement_speed = 8.0
    mouse_sens = 0.12
    last_time = glfw.get_time()

    def cursor_pos(window, xpos, ypos):
        nonlocal lastX, lastY, yaw, pitch, cam_front, first_mouse
        if first_mouse:
            lastX = xpos; lastY = ypos; first_mouse = False
        xoffset = xpos - lastX; yoffset = lastY - ypos
        lastX = xpos; lastY = ypos
        xoffset *= mouse_sens; yoffset *= mouse_sens
        yaw += xoffset; pitch += yoffset
        if pitch > 89.0: pitch = 89.0
        if pitch < -89.0: pitch = -89.0
        front = glm.vec3(
            np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
            np.sin(np.radians(pitch)),
            np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
        )
        cam_front = glm.normalize(front)

    glfw.set_cursor_pos_callback(window, cursor_pos)

    # свет - изменена позиция для правильных теней
    light_pos = glm.vec3(-10.0, 20.0, 5.0)

    # позиции домиков: одна линия по оси X, одинаковая z
    line_z = 0.0
    pos_patrick = glm.vec3(-8.0, 0.0, line_z)
    pos_squid = glm.vec3(0.0, 0.0, line_z)
    pos_sponge = glm.vec3(8.0, 0.0, line_z)

    # вспомогательная функция - обработка управления WASD/Space/Ctrl
    def process_movement(delta):
        nonlocal cam_pos
        speed = movement_speed * delta
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            cam_pos += cam_front * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            cam_pos -= cam_front * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            cam_pos -= glm.normalize(glm.cross(cam_front, cam_up)) * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            cam_pos += glm.normalize(glm.cross(cam_front, cam_up)) * speed
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            cam_pos += cam_up * speed
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            cam_pos -= cam_up * speed

    # рендер цикл
    while not glfw.window_should_close(window):
        current = glfw.get_time()
        delta = current - last_time
        last_time = current
        glfw.poll_events()
        process_movement(delta)
        w, h = glfw.get_framebuffer_size(window)

        # 1) рендер теневой карты
        near_plane, far_plane = 1.0, 120.0
        light_proj = glm.ortho(-60.0,60.0,-60.0,60.0, near_plane, far_plane)
        light_view = glm.lookAt(light_pos, glm.vec3(0.0,0.0,0.0), glm.vec3(0.0,1.0,0.0))
        light_space = light_proj * light_view
        glViewport(0,0,SHADOW_W,SHADOW_H)
        glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glUseProgram(depth_prog)
        glUniformMatrix4fv(glGetUniformLocation(depth_prog, 'lightSpaceMatrix'), 1, GL_FALSE, glm.value_ptr(light_space))

        def render_depth():
            model = glm.translate(glm.mat4(1.0), glm.vec3(0.0,-0.01,0.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model))
            glBindVertexArray(plane_vao); glDrawArrays(GL_TRIANGLES, 0, plane_count)
            # Patrick
            model_p = glm.translate(glm.mat4(1.0), pos_patrick + glm.vec3(0.0,0.6,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,1.0,1.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_p))
            glBindVertexArray(sph_vao); glDrawArrays(GL_TRIANGLES,0,sph_count)
            # Squidward - увеличенный цилиндр
            model_sq = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.0,0.0,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,4.0,1.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_sq))
            glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)
            # Sponge (half ellipsoid) - ИСПРАВЛЕНО: добавлен в теневой рендер
            model_sb = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0,0.6,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.6,2.0,1.2))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_sb))
            glBindVertexArray(sph_vao); glDrawArrays(GL_TRIANGLES,0,sph_count)

        render_depth()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 2) основной рендер
        glViewport(0,0,w,h)
        glClearColor(0.6,0.85,0.92,1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(prog)

        # камера и матрицы
        view = glm.lookAt(cam_pos, cam_pos + cam_front, cam_up)
        proj = glm.perspective(glm.radians(60.0), w/h if h>0 else 1.0, 0.1, 200.0)
        glUniformMatrix4fv(glGetUniformLocation(prog,'view'),1,GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(prog,'projection'),1,GL_FALSE, glm.value_ptr(proj))
        glUniform3fv(glGetUniformLocation(prog,'lightPos'),1, glm.value_ptr(light_pos))
        glUniform3fv(glGetUniformLocation(prog,'viewPos'),1, glm.value_ptr(cam_pos))
        glUniformMatrix4fv(glGetUniformLocation(prog,'lightSpaceMatrix'),1,GL_FALSE, glm.value_ptr(light_space))

        # bind shadow map
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, depth_map); glUniform1i(glGetUniformLocation(prog,'shadowMap'), 1)

        def draw_textured(vao, count, tex, model, shininess=32.0):
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex); glUniform1i(glGetUniformLocation(prog,'texture_diffuse1'), 0)
            glUniform1f(glGetUniformLocation(prog,'materialShininess'), shininess)
            glUniformMatrix4fv(glGetUniformLocation(prog,'model'),1,GL_FALSE, glm.value_ptr(model))
            glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, count)

        # Небо (боковые стороны с background.png)
        sky_model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.0, 1.0, 1.0))
        draw_textured(skybox_sides_vao, skybox_sides_count, tex_background, sky_model, 128.0)
        
        # Небо (верхняя часть с sky.png)
        draw_textured(skybox_top_vao, skybox_top_count, tex_sky, sky_model, 128.0)

        # plane (sand)
        draw_textured(plane_vao, plane_count, tex_sand, glm.translate(glm.mat4(1.0), glm.vec3(0.0,-0.01,0.0)), 8.0)

        # Дороги
        # Главная дорога
        road_main = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 8.0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.1, 3.0))
        draw_textured(cube_vao, cube_count, tex_road, road_main, 16.0)
        
        # Дорожка к дому Патрика
        road_patrick = glm.translate(glm.mat4(1.0), glm.vec3(-8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.1, 1.0))
        draw_textured(cube_vao, cube_count, tex_road, road_patrick, 16.0)
        
        # Дорожка к дому Спанчбоба
        road_sponge = glm.translate(glm.mat4(1.0), glm.vec3(8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.1, 1.0))
        draw_textured(cube_vao, cube_count, tex_road, road_sponge, 16.0)

        # Дорожка к дому Сквидварда
        road_squid1 = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 2.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
        draw_textured(cube_vao, cube_count, tex_road, road_squid1, 16.0)
        road_squid2 = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 3.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
        draw_textured(cube_vao, cube_count, tex_road, road_squid2, 16.0)
        road_squid3 = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 4.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
        draw_textured(cube_vao, cube_count, tex_road, road_squid3, 16.0)
        road_squid4 = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 5.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
        draw_textured(cube_vao, cube_count, tex_road, road_squid4, 16.0)
        road_squid5 = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 6.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
        draw_textured(cube_vao, cube_count, tex_road, road_squid5, 16.0)

        # Patrick (rock) - слева
        model_patrick = glm.translate(glm.mat4(1.0), pos_patrick + glm.vec3(0.0,0.0,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,1.6,1.8))
        draw_textured(sph_vao, sph_count, tex_rock, model_patrick, 6.0)

        # Флюгер (буква T) над домом Патрика
        # Вертикальная часть буквы T
        weathervane_vertical = model_for_box((pos_patrick.x, pos_patrick.y + 1.5, pos_patrick.z), (0.1, 0.75, 0.1))
        draw_textured(cube_vao, cube_count, tex_wood, weathervane_vertical, 24.0)
        
        # Горизонтальная часть буквы T
        weathervane_horizontal = model_for_box((pos_patrick.x, pos_patrick.y + 1.92, pos_patrick.z), (0.6, 0.1, 0.1))
        draw_textured(cube_vao, cube_count, tex_wood, weathervane_horizontal, 24.0)

        # Squidward (центр): увеличенный цилиндр + уши + монобровь + нос + крыша + окна + дверь
        model_sq_base = glm.translate(glm.mat4(1.0), pos_squid) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,7.0,1.8))
        draw_textured(cyl_vao, cyl_count, tex_squid, model_sq_base, 12.0)
        
        # Крыша (диск)
        roof_squid = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.0,3.5,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,1.8,1.8))
        draw_textured(disk_vao, disk_count, tex_squid, roof_squid, 12.0)
        
        # уши (слева/справа)
        ear_right = model_for_box((pos_squid.x + 1.0, pos_squid.y + 2.0, pos_squid.z), (0.2,0.8,0.2))
        ear_left  = model_for_box((pos_squid.x - 1.0, pos_squid.y + 2.0, pos_squid.z), (0.2,0.8,0.2))
        draw_textured(cube_vao, cube_count, tex_squid, ear_right, 12.0)
        draw_textured(cube_vao, cube_count, tex_squid, ear_left, 12.0)
        
        # монобровь
        brow = model_for_box((pos_squid.x, pos_squid.y + 2.6, pos_squid.z + 0.8), (1.4,0.25,0.25))
        draw_textured(cube_vao, cube_count, tex_squid, brow, 12.0)
        
        # нос
        nose = model_for_box((pos_squid.x, pos_squid.y + 2.0, pos_squid.z + 0.9), (0.4,1.0,0.4))
        draw_textured(cube_vao, cube_count, tex_squid, nose, 12.0)
        
        # Окна
        window_left = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(-0.45, 2.2, 0.9)) * glm.scale(glm.mat4(1.0), glm.vec3(0.5,0.5,0.5))
        window_right = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.45, 2.2, 0.9)) * glm.scale(glm.mat4(1.0), glm.vec3(0.5,0.5,0.5))
        
        # Рамки окон (объемные)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_left, 64.0)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_right, 64.0)
        
        # Стекла окон (немного смещены назад относительно рамок)
        window_glass_left = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(-0.45, 2.2, 0.9)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.35,0.35,0.35))
        window_glass_right = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.45, 2.2, 0.9)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.35,0.35,0.35))
        
        draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_left, 64.0)
        draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_right, 64.0)
        
        # Дверь (деревянная)
        door_squid = model_for_box((pos_squid.x, pos_squid.y + 0.5, pos_squid.z + 0.88), (0.6,1.2,0.05))
        draw_textured(cube_vao, cube_count, tex_wood, door_squid, 24.0)

        # SpongeBob (ананас) — половина эллипсоида + листья сверху + окна + дверь
        model_sponge = glm.translate(glm.mat4(1.0), pos_sponge) * glm.scale(glm.mat4(1.0), glm.vec3(1.5,2.8,1.5))
        draw_textured(sph_vao, sph_count, tex_pine, model_sponge, 32.0)
        
        # листья: наклоненные для естественного вида
        leaf1 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, 3.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.8,0.6))
        leaf2 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(-0.25, 2.9, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(45.0), glm.vec3(0,1,1)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.7,0.6))
        leaf3 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.25, 2.9, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(-45.0), glm.vec3(0,1,1)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.7,0.6))
        draw_textured(cube_vao, cube_count, tex_leaf, leaf1, 24.0)
        draw_textured(cube_vao, cube_count, tex_leaf, leaf2, 24.0)
        draw_textured(cube_vao, cube_count, tex_leaf, leaf3, 24.0)
        
        # Окна с правильной ориентацией на сферической поверхности
        # Для сферического дома окна должны быть направлены по нормали к поверхности
        window_sponge_left = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(-0.8, 2.0, 0.75))  * glm.rotate(glm.mat4(1.0), glm.radians(45.0), glm.vec3(0,0,1)) * glm.rotate(glm.mat4(1.0), glm.radians(-50.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6,0.6,0.6))
        window_sponge_right = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.7, 1.3, 1.15)) * glm.rotate(glm.mat4(1.0), glm.radians(-65.0), glm.vec3(0,0,1)) * glm.rotate(glm.mat4(1.0), glm.radians(-33.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6,0.6,0.6))
        
        # Рамки окон (объемные)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_sponge_left, 64.0)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_sponge_right, 64.0)
        w
        # Стекла окон (немного смещены назад)
        window_glass_sponge_left = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(-0.8, 2.0, 0.75)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.rotate(glm.mat4(1.0), glm.radians(45.0), glm.vec3(0,0,1)) * glm.rotate(glm.mat4(1.0), glm.radians(-33.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6,0.6,0.6))
        window_glass_sponge_right = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.7, 1.3, 1.15)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(0,0,1)) * glm.rotate(glm.mat4(1.0), glm.radians(-10.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6,0.6,0.6))
        
        draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_sponge_left, 64.0)
        draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_sponge_right, 64.0)
        
        # Дверь (металлическая)
        door_sponge = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, -0.25, 1.45)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 1.3, 0.1))
        draw_textured(sph_vao, sph_count, tex_metal, door_sponge, 32.0)
        
        # Рамка вокруг двери Спанчбоба (изогнутая, повторяет форму полуэллипса)
        door_frame_model = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, 0.0, 1.45)) * glm.scale(glm.mat4(1.0), glm.vec3(1.0, 2.2, 0.8))
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, door_frame_model, 64.0)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == '__main__':
    main()