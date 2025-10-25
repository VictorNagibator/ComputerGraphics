from OpenGL.GL import *
import glfw
import numpy as np
from PIL import Image
import ctypes
from pyglm import glm
import os
import sys
import random
import math

# ---------- Шейдеры (GLSL) ----------
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
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x=-2;x<=2;x++){
        for(int y=-2;y<=2;y++){
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y)*texelSize).r;
            shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;
        }
    }
    shadow /= 25.0;
    return shadow;
}

void main(){
    vec3 color = texture(texture_diffuse1, TexCoord).rgb;
    vec3 normal = normalize(Normal);
    vec3 lightColor = vec3(1.0);
    vec3 ambient = 0.3 * color;
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
    vec3 specular = spec * lightColor * 0.3;
    float shadow = ShadowCalculation(FragPosLightSpace, normal, lightDir);
    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(lighting, 1.0);
}
"""

FLOWER_VERT_SHADER = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTex;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    TexCoord = aTex;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FLOWER_FRAG_SHADER = """#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture_diffuse1;

void main(){
    vec4 texColor = texture(texture_diffuse1, TexCoord);
    if(texColor.a < 0.1)
        discard;
    FragColor = texColor;
}
"""

SUN_VERT_SHADER = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTex;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTex;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

SUN_FRAG_SHADER = """#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform vec3 sunPos;
uniform vec3 viewPos;

void main(){
    // Основной цвет солнца - ярко-желтый
    vec3 sunColor = vec3(1.0, 0.9, 0.4);
    
    // Вычисляем нормаль и направление к камере
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 lightDir = normalize(sunPos - FragPos);
    
    // Яркое ядро солнца
    float coreIntensity = 2.0;
    vec3 coreColor = sunColor * coreIntensity;
    
    // Эффект блика - когда смотрим почти прямо на солнце
    float spec = pow(max(dot(viewDir, lightDir), 0.0), 128.0);
    vec3 glare = vec3(1.0, 1.0, 0.8) * spec * 3.0;
    
    // Рассеянное свечение вокруг солнца
    float glow = 1.0 - dot(norm, viewDir);
    glow = pow(glow, 2.0);
    vec3 glowColor = vec3(1.0, 0.7, 0.3) * glow * 1.5;
    
    // Комбинируем все эффекты
    vec3 finalColor = coreColor + glare + glowColor;
    
    // Усиливаем интенсивность
    finalColor *= 2.0;
    
    FragColor = vec4(finalColor, 1.0);
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

# ---------- Помощники для шейдеров ----------

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

# ---------- Примитивы ----------

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
    verts = []
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        p0_outer_front = (outer_radius*np.cos(a0), outer_radius*np.sin(a0), thickness/2)
        p1_outer_front = (outer_radius*np.cos(a1), outer_radius*np.sin(a1), thickness/2)
        p0_outer_back = (outer_radius*np.cos(a0), outer_radius*np.sin(a0), -thickness/2)
        p1_outer_back = (outer_radius*np.cos(a1), outer_radius*np.sin(a1), -thickness/2)
        p0_inner_front = (inner_radius*np.cos(a0), inner_radius*np.sin(a0), thickness/2)
        p1_inner_front = (inner_radius*np.cos(a1), inner_radius*np.sin(a1), thickness/2)
        p0_inner_back = (inner_radius*np.cos(a0), inner_radius*np.sin(a0), -thickness/2)
        p1_inner_back = (inner_radius*np.cos(a1), inner_radius*np.sin(a1), -thickness/2)
        n_outer = (np.cos(a0), np.sin(a0), 0)
        n_inner = (-np.cos(a0), -np.sin(a0), 0)
        n_top = (0, 0, 1)
        n_bottom = (0, 0, -1)
        verts.extend([*p0_outer_front, *n_outer, i/segments, 0])
        verts.extend([*p0_outer_back, *n_outer, i/segments, 1])
        verts.extend([*p1_outer_front, *n_outer, (i+1)/segments, 0])
        verts.extend([*p1_outer_front, *n_outer, (i+1)/segments, 0])
        verts.extend([*p0_outer_back, *n_outer, i/segments, 1])
        verts.extend([*p1_outer_back, *n_outer, (i+1)/segments, 1])
        verts.extend([*p0_inner_front, *n_inner, i/segments, 0])
        verts.extend([*p1_inner_front, *n_inner, (i+1)/segments, 0])
        verts.extend([*p0_inner_back, *n_inner, i/segments, 1])
        verts.extend([*p1_inner_front, *n_inner, (i+1)/segments, 0])
        verts.extend([*p1_inner_back, *n_inner, (i+1)/segments, 1])
        verts.extend([*p0_inner_back, *n_inner, i/segments, 1])
        verts.extend([*p0_outer_front, *n_top, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p1_outer_front, *n_top, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p0_inner_front, *n_top, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_outer_front, *n_top, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p1_inner_front, *n_top, 0.5+0.45*np.cos(a1), 0.5+0.45*np.sin(a1)])
        verts.extend([*p0_inner_front, *n_top, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p0_outer_back, *n_bottom, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p0_inner_back, *n_bottom, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_outer_back, *n_bottom, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p1_outer_back, *n_bottom, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
        verts.extend([*p0_inner_back, *n_bottom, 0.5+0.45*np.cos(a0), 0.5+0.45*np.sin(a0)])
        verts.extend([*p1_inner_back, *n_bottom, 0.5+0.45*np.cos(a1), 0.5+0.45*np.sin(a1)])
    return np.array(verts, dtype=np.float32)

def create_flower_quad(size=1.0):
    verts = []
    s = size / 2.0
    coords = [
        (-s, -s, 0.0), (s, s, 0.0), (s, -s, 0.0),
        (s, s, 0.0), (-s, -s, 0.0), (-s, s, 0.0)
    ]
    tex_coords = [
        (0.0, 0.0), (1.0, 1.0), (1.0, 0.0),
        (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)
    ]
    normal = (0.0, 0.0, 1.0)
    for i in range(6):
        verts.extend([*coords[i], *normal, *tex_coords[i]])
    return np.array(verts, dtype=np.float32)

def create_curved_road(length=40.0, width=3.0, curve_radius=25.0, segments=64):
    verts = []
    start_x = -20.0
    start_z = 8.0
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        angle0 = t0 * (math.pi / 2)
        angle1 = t1 * (math.pi / 2)
        x0 = start_x - curve_radius * math.sin(angle0)
        z0 = start_z + curve_radius * (1 - math.cos(angle0))
        x1 = start_x - curve_radius * math.sin(angle1)
        z1 = start_z + curve_radius * (1 - math.cos(angle1))
        dx0 = -curve_radius * math.cos(angle0)
        dz0 = curve_radius * math.sin(angle0)
        dx1 = -curve_radius * math.cos(angle1)
        dz1 = curve_radius * math.sin(angle1)
        length0 = math.sqrt(dx0*dx0 + dz0*dz0)
        length1 = math.sqrt(dx1*dx1 + dz1*dz1)
        if length0 > 0:
            dx0 /= length0
            dz0 /= length0
        if length1 > 0:
            dx1 /= length1
            dz1 /= length1
        px0 = -dz0
        pz0 = dx0
        px1 = -dz1
        pz1 = dx1
        half_width = width / 2.0
        left_x0 = x0 + px0 * half_width
        left_z0 = z0 + pz0 * half_width
        left_x1 = x1 + px1 * half_width
        left_z1 = z1 + pz1 * half_width
        right_x0 = x0 - px0 * half_width
        right_z0 = z0 - pz0 * half_width
        right_x1 = x1 - px1 * half_width
        right_z1 = z1 - pz1 * half_width
        normal = (0.0, 1.0, 0.0)
        height = 0.005 
        u0 = t0 * 10.0
        u1 = t1 * 10.0
        v_left = 0.0
        v_right = 1.0
        verts.extend([left_x0, height, left_z0, *normal, u0, v_left])
        verts.extend([right_x1, height, right_z1, *normal, u1, v_right])
        verts.extend([right_x0, height, right_z0, *normal, u0, v_right])
        verts.extend([left_x0, height, left_z0, *normal, u0, v_left])
        verts.extend([left_x1, height, left_z1, *normal, u1, v_left])
        verts.extend([right_x1, height, right_z1, *normal, u1, v_right])
    return np.array(verts, dtype=np.float32)

def create_rectangular_window(width=1.0, height=1.0, frame_thickness=0.05):
    verts = []
    half_w = width / 2.0
    half_h = height / 2.0
    half_t = frame_thickness / 2.0
    
    # Вертикальные части рамки
    # Левая
    verts.extend([-half_w, -half_h, half_t, -1, 0, 0, 0, 0])
    verts.extend([-half_w, half_h, half_t, -1, 0, 0, 1, 1])
    verts.extend([-half_w, half_h, -half_t, -1, 0, 0, 1, 0])
    verts.extend([-half_w, -half_h, half_t, -1, 0, 0, 0, 0])
    verts.extend([-half_w, half_h, -half_t, -1, 0, 0, 1, 0])
    verts.extend([-half_w, -half_h, -half_t, -1, 0, 0, 0, 1])
    
    # Правая
    verts.extend([half_w, -half_h, half_t, 1, 0, 0, 0, 0])
    verts.extend([half_w, half_h, -half_t, 1, 0, 0, 1, 1])
    verts.extend([half_w, half_h, half_t, 1, 0, 0, 1, 0])
    verts.extend([half_w, -half_h, half_t, 1, 0, 0, 0, 0])
    verts.extend([half_w, -half_h, -half_t, 1, 0, 0, 0, 1])
    verts.extend([half_w, half_h, -half_t, 1, 0, 0, 1, 1])
    
    # Горизонтальные части рамки
    # Верхняя
    verts.extend([-half_w, half_h, half_t, 0, 1, 0, 0, 0])
    verts.extend([half_w, half_h, half_t, 0, 1, 0, 1, 1])
    verts.extend([half_w, half_h, -half_t, 0, 1, 0, 1, 0])
    verts.extend([-half_w, half_h, half_t, 0, 1, 0, 0, 0])
    verts.extend([half_w, half_h, -half_t, 0, 1, 0, 1, 0])
    verts.extend([-half_w, half_h, -half_t, 0, 1, 0, 0, 1])
    
    # Нижняя
    verts.extend([-half_w, -half_h, half_t, 0, -1, 0, 0, 0])
    verts.extend([half_w, -half_h, -half_t, 0, -1, 0, 1, 1])
    verts.extend([half_w, -half_h, half_t, 0, -1, 0, 1, 0])
    verts.extend([-half_w, -half_h, half_t, 0, -1, 0, 0, 0])
    verts.extend([-half_w, -half_h, -half_t, 0, -1, 0, 0, 1])
    verts.extend([half_w, -half_h, -half_t, 0, -1, 0, 1, 1])
    
    return np.array(verts, dtype=np.float32)

def create_half_cylinder_with_caps(segments=64):
    """Создает полуцилиндр с половинами верхнего и нижнего оснований"""
    verts = []
    
    # Основная изогнутая поверхность (половина цилиндра)
    for i in range(segments//2):
        a0 = np.pi * i/segments * 2
        a1 = np.pi * (i+1)/segments * 2
        
        # Точки для изогнутой поверхности
        p0_bottom = (0.5*np.cos(a0), -0.5, 0.5*np.sin(a0))
        p1_bottom = (0.5*np.cos(a1), -0.5, 0.5*np.sin(a1))
        p0_top = (0.5*np.cos(a0), 0.5, 0.5*np.sin(a0))
        p1_top = (0.5*np.cos(a1), 0.5, 0.5*np.sin(a1))
        
        # Нормали для изогнутой поверхности
        n0 = (np.cos(a0), 0, np.sin(a0))
        n1 = (np.cos(a1), 0, np.sin(a1))
        
        # Изогнутая поверхность
        verts.extend([*p0_bottom, *n0, i/segments, 0])
        verts.extend([*p0_top, *n0, i/segments, 1])
        verts.extend([*p1_bottom, *n1, (i+1)/segments, 0])
        verts.extend([*p0_top, *n0, i/segments, 1])
        verts.extend([*p1_top, *n1, (i+1)/segments, 1])
        verts.extend([*p1_bottom, *n1, (i+1)/segments, 0])
    
    # Нижнее основание (полукруг)
    center_bottom = (0.0, -0.5, 0.0)
    normal_bottom = (0.0, -1.0, 0.0)
    for i in range(segments//2):
        a0 = np.pi * i/segments * 2
        a1 = np.pi * (i+1)/segments * 2
        
        p0 = (0.5*np.cos(a0), -0.5, 0.5*np.sin(a0))
        p1 = (0.5*np.cos(a1), -0.5, 0.5*np.sin(a1))
        
        verts.extend([*center_bottom, *normal_bottom, 0.5, 0.5])
        verts.extend([*p0, *normal_bottom, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p1, *normal_bottom, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
    
    # Верхнее основание (полукруг)
    center_top = (0.0, 0.5, 0.0)
    normal_top = (0.0, 1.0, 0.0)
    for i in range(segments//2):
        a0 = np.pi * i/segments * 2
        a1 = np.pi * (i+1)/segments * 2
        
        p0 = (0.5*np.cos(a0), 0.5, 0.5*np.sin(a0))
        p1 = (0.5*np.cos(a1), 0.5, 0.5*np.sin(a1))
        
        verts.extend([*center_top, *normal_top, 0.5, 0.5])
        verts.extend([*p0, *normal_top, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)])
        verts.extend([*p1, *normal_top, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)])
    
    return np.array(verts, dtype=np.float32)

def create_sign_post(height=4.0, radius=0.05):
    """Создает столб для вывески"""
    verts = []
    segments = 16
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        p0 = (radius*np.cos(a0), -height/2, radius*np.sin(a0))
        p1 = (radius*np.cos(a1), -height/2, radius*np.sin(a1))
        p2 = (radius*np.cos(a0), height/2, radius*np.sin(a0))
        p3 = (radius*np.cos(a1), height/2, radius*np.sin(a1))
        n0 = (np.cos(a0),0,np.sin(a0))
        n1 = (np.cos(a1),0,np.sin(a1))
        verts.extend([*p0,*n0, i/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p3,*n1, (i+1)/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
    return np.array(verts, dtype=np.float32)

def create_shell_sign(width=2.0, height=1.0, curvature=0.3):
    """Создает вывеску в форме ракушки"""
    verts = []
    segments = 32
    for i in range(segments):
        u0 = i / segments
        u1 = (i + 1) / segments
        for j in range(segments//2):
            v0 = j / (segments//2)
            v1 = (j + 1) / (segments//2)
            
            # Параметрическая поверхность для ракушки
            x0 = (u0 - 0.5) * width
            z0 = (v0 - 0.5) * height
            y0 = curvature * np.sin(u0 * np.pi) * np.sin(v0 * np.pi)
            
            x1 = (u1 - 0.5) * width
            z1 = (v0 - 0.5) * height
            y1 = curvature * np.sin(u1 * np.pi) * np.sin(v0 * np.pi)
            
            x2 = (u0 - 0.5) * width
            z2 = (v1 - 0.5) * height
            y2 = curvature * np.sin(u0 * np.pi) * np.sin(v1 * np.pi)
            
            x3 = (u1 - 0.5) * width
            z3 = (v1 - 0.5) * height
            y3 = curvature * np.sin(u1 * np.pi) * np.sin(v1 * np.pi)
            
            # Нормали (упрощенные)
            normal = (0, 1, 0)
            
            verts.extend([x0, y0, z0, *normal, u0, v0])
            verts.extend([x1, y1, z1, *normal, u1, v0])
            verts.extend([x2, y2, z2, *normal, u0, v1])
            
            verts.extend([x1, y1, z1, *normal, u1, v0])
            verts.extend([x3, y3, z3, *normal, u1, v1])
            verts.extend([x2, y2, z2, *normal, u0, v1])
    
    return np.array(verts, dtype=np.float32)

def create_flag(width=0.8, height=0.5):
    verts = []
    # Прямоугольный флажок
    coords = [
        (0, 0, 0), (width, 0, 0), (width, height, 0),
        (0, 0, 0), (width, height, 0), (0, height, 0)
    ]
    tex_coords = [
        (0, 0), (1, 0), (1, 1),
        (0, 0), (1, 1), (0, 1)
    ]
    normal = (0, 0, 1)
    for i in range(6):
        verts.extend([*coords[i], *normal, *tex_coords[i]])
    return np.array(verts, dtype=np.float32)


def create_rope(length=10.0, segments=32, sag=0.8):
    """Создает веревку для гирлянды"""
    verts = []
    radius = 0.06
    half_len = length / 2
    
    for i in range(segments):
        a0 = 2*np.pi*i/segments
        a1 = 2*np.pi*(i+1)/segments
        
        # Параболическое провисание
        x0 = -half_len + (i/segments)*length
        x1 = -half_len + ((i+1)/segments)*length
        
        # Парабола: y = a*x^2 + c, где a = -4*sag/length^2
        a = -4 * sag / (length * length)
        y0 = a * x0 * x0 + sag
        y1 = a * x1 * x1 + sag
        
        # Точки для окружности веревки
        p0 = (x0, y0, radius*np.cos(a0))
        p1 = (x1, y1, radius*np.cos(a1))
        p2 = (x0, y0, radius*np.sin(a0))
        p3 = (x1, y1, radius*np.sin(a1))
        
        # Нормали (аппроксимация)
        tangent_x = x1 - x0
        tangent_y = y1 - y0
        tangent_len = np.sqrt(tangent_x*tangent_x + tangent_y*tangent_y)
        if tangent_len > 0:
            tangent_x /= tangent_len
            tangent_y /= tangent_len
        normal_x = -tangent_y
        normal_y = tangent_x
        
        n0 = (normal_x*np.cos(a0), normal_y*np.cos(a0), np.sin(a0))
        n1 = (normal_x*np.cos(a1), normal_y*np.cos(a1), np.sin(a1))
        
        verts.extend([*p0,*n0, i/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
        verts.extend([*p2,*n0, i/segments,1])
        verts.extend([*p3,*n1, (i+1)/segments,1])
        verts.extend([*p1,*n1, (i+1)/segments,0])
    
    return np.array(verts, dtype=np.float32)

def get_parabolic_position(x, length=10.0, sag=0.8):
    """Вычисляет позицию на параболической веревке"""
    half_len = length / 2
    a = -4 * sag / (length * length)
    y = a * x * x + sag
    return y

def load_texture(path):
    img = Image.open(path).convert('RGBA')
    img_data = np.array(img)[::-1]
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

# ---------- GL буферы/VAO ----------

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

def model_for_box(center, size):
    return glm.translate(glm.mat4(1.0), glm.vec3(*center)) * glm.scale(glm.mat4(1.0), glm.vec3(*size))

def main():
    if not glfw.init():
        print('GLFW init failed')
        return
    width, height = 1280, 720
    window = glfw.create_window(width, height, 'Лабораторная работа 4 - Бикини Боттом', None, None)
    if not window:
        print('Window creation failed')
        glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    prog = link_program(VERT_SHADER, FRAG_SHADER)
    depth_prog = link_program(DEPTH_VS, DEPTH_FS)
    flower_prog = link_program(FLOWER_VERT_SHADER, FLOWER_FRAG_SHADER)
    sun_prog = link_program(SUN_VERT_SHADER, SUN_FRAG_SHADER)

    # создать меши
    plane_data = create_plane(80.0, uv_scale=30.0)
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
    flower_data = create_flower_quad(8.0)
    flower_vao, _, flower_count = make_vao(flower_data)
    curved_road_data = create_curved_road(length=40.0, width=3.0, curve_radius=25.0, segments=64)
    curved_road_vao, _, curved_road_count = make_vao(curved_road_data)
    rectangular_window_data = create_rectangular_window(1.0, 1.0, 0.08)
    rectangular_window_vao, _, rectangular_window_count = make_vao(rectangular_window_data)
    half_cyl_data = create_half_cylinder_with_caps(64)
    half_cyl_vao, _, half_cyl_count = make_vao(half_cyl_data)
    sign_post_data = create_sign_post()
    sign_post_vao, _, sign_post_count = make_vao(sign_post_data)
    shell_sign_data = create_shell_sign()
    shell_sign_vao, _, shell_sign_count = make_vao(shell_sign_data)
    flag_data = create_flag()
    flag_vao, _, flag_count = make_vao(flag_data)
    rope_data = create_rope(length=10.0, segments=64, sag=1.2)
    rope_vao, _, rope_count = make_vao(rope_data)

    # текстуры
    texdir = os.path.join(os.path.dirname(__file__), 'textures')
    tex_sand = load_texture(os.path.join(texdir, 'sand.png'))
    tex_flowers = [
        load_texture(os.path.join(texdir, 'green_flower.png')), 
        load_texture(os.path.join(texdir, 'pink_flower.png')),
        load_texture(os.path.join(texdir, 'blue_flower.png')),
        load_texture(os.path.join(texdir, 'yellow_flower.png'))
    ]
    tex_pine = load_texture(os.path.join(texdir, 'pineapple.png'))
    tex_rock = load_texture(os.path.join(texdir, 'rock.png'))
    tex_squid = load_texture(os.path.join(texdir, 'squidward.png'))
    tex_leaf = load_texture(os.path.join(texdir, 'leaf.png'))
    tex_road = load_texture(os.path.join(texdir, 'road.png'))
    tex_wood = load_texture(os.path.join(texdir, 'wood.png'))
    tex_metal = load_texture(os.path.join(texdir, 'metal.png'))
    tex_metal_house = load_texture(os.path.join(texdir, 'metal_house.png'))
    tex_krusty_krab = load_texture(os.path.join(texdir, 'krusty_krab.png'))
    
    def create_color_texture(r, g, b, a=1.0):
        data = np.array([[int(r*255), int(g*255), int(b*255), int(a*255)]], dtype=np.uint8)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex
    
    tex_window_blue = create_color_texture(0.5, 0.7, 1.0)
    tex_window_frame = create_color_texture(0.2, 0.3, 0.8)
    tex_glass_door = create_color_texture(0.7, 0.9, 1.0, 0.8)  # Стеклянная дверь
    tex_gold = create_color_texture(0.9, 0.7, 0.1)  # Золотистый цвет
    tex_shell = create_color_texture(1.0, 0.9, 0.8)     # Светлый цвет ракушки

    # Создаем текстуру с текстом "The Krusty Krab"
    def create_text_texture(text, width=256, height=64, font_size=32):
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Создаем изображение
            img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Пытаемся использовать шрифт Arial, если нет - используем стандартный
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Получаем размеры текста
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Центрируем текст
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Рисуем текст красным цветом
            draw.text((x, y), text, font=font, fill=(255, 0, 0, 255))
            
            # Конвертируем в numpy массив
            img_data = np.array(img)[::-1]
            
            # Создаем текстуру
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            return tex
        except Exception as e:
            print(f"Failed to create text texture: {e}")
            # Возвращаем простую красную текстуру в случае ошибки
            return create_color_texture(1.0, 0.0, 0.0)
    
    tex_sign_text = create_text_texture("        The\nKRUSTY KRAB", width=512, height=128, font_size=48)

    # Цвета флажков: красный, желтый, синий (чередуются)
    flag_colors = [
        create_color_texture(1.0, 0.0, 0.0),   # Красный
        create_color_texture(1.0, 1.0, 0.0),   # Желтый
        create_color_texture(0.0, 0.0, 1.0),   # Синий
        create_color_texture(1.0, 0.0, 0.0),   # Красный
        create_color_texture(1.0, 1.0, 0.0),   # Желтый
        create_color_texture(0.0, 0.0, 1.0),   # Синий
    ]
    
    tex_rope = create_color_texture(0.5, 0.4, 0.3)  # Цвет веревки

    # теневой фреймбуфер
    SHADOW_W, SHADOW_H = 4096, 4096 
    depth_map_fbo = glGenFramebuffers(1)
    depth_map = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_map)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_W, SHADOW_H, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
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
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # камера
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

    # свет - позиция для солнца в небе
    light_pos = glm.vec3(40.0, 80.0, 40.0)

    # позиции домиков
    line_z = 0.0
    pos_patrick = glm.vec3(-8.0, 0.0, line_z)
    pos_squid = glm.vec3(0.0, 0.0, line_z)
    pos_sponge = glm.vec3(8.0, 0.0, line_z)
    pos_krusty_krab = glm.vec3(50.0, 0.0, line_z)
    
    # Позиции для цветков в небе
    flower_positions = []
    for _ in range(30):
        x = random.uniform(-80, 80)
        y = random.uniform(35, 45)
        z = random.uniform(-80, 80)
        flower_type = random.randint(0, len(tex_flowers) - 1)
        flower_positions.append((glm.vec3(x, y, z), flower_type))
        
    # Позиции для домиков обычных жителей
    house_positions = [glm.vec3(-35, 0.0, 9.5),
                       glm.vec3(-36, 0.0, 18.0),
                       glm.vec3(-44, 0.0, 18.0),
                       glm.vec3(-48, 0.0, 32.0),
                       glm.vec3(-41, 0.0, 30.0),
                       glm.vec3(-47, 0.0, 25.0),
                       ]
    
    # случайная высота домиков
    random_height_bonuses = [random.randint(-2, 2) for _ in range(len(house_positions))]

    # случайная текстура домика
    random_texture_for_houses = [tex_metal_house if random.randint(0,1) == 1 else tex_metal for _ in range(len(house_positions))]
    
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
        # Проверка на закрытие по ESC
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            
        current = glfw.get_time()
        delta = current - last_time
        last_time = current
        glfw.poll_events()
        process_movement(delta)
        w, h = glfw.get_framebuffer_size(window)

        # 1) рендер теневой карты
        near_plane, far_plane = 1.0, 150.0
        light_proj = glm.ortho(-70.0,70.0,-70.0,70.0, near_plane, far_plane)
        light_view = glm.lookAt(light_pos, glm.vec3(0.0,0.0,0.0), glm.vec3(0.0,1.0,0.0))
        light_space = light_proj * light_view
        glViewport(0,0,SHADOW_W,SHADOW_H)
        glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glUseProgram(depth_prog)
        glUniformMatrix4fv(glGetUniformLocation(depth_prog, 'lightSpaceMatrix'), 1, GL_FALSE, glm.value_ptr(light_space))

        def render_depth():
            # Плоскость
            model = glm.translate(glm.mat4(1.0), glm.vec3(0.0,-0.01,0.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model))
            glBindVertexArray(plane_vao); glDrawArrays(GL_TRIANGLES, 0, plane_count)
            
            # Дороги
            road_main = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 8.0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_main))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            curved_road_model = glm.mat4(1.0)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(curved_road_model))
            glBindVertexArray(curved_road_vao); glDrawArrays(GL_TRIANGLES,0,curved_road_count)
            
            road_next = glm.translate(glm.mat4(1.0), glm.vec3(-45.0, 0.0, 53.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_next))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            road_to_krusty_krabs = glm.translate(glm.mat4(1.0), glm.vec3(40.0, 0.0, 8.0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_to_krusty_krabs))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Дорожки к домам
            road_patrick = glm.translate(glm.mat4(1.0), glm.vec3(-8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.01, 1.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_patrick))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            road_sponge = glm.translate(glm.mat4(1.0), glm.vec3(8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.01, 1.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_sponge))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Дорожка к дому Сквидварда
            for z in [2.0, 3.0, 4.0, 5.0, 6.0]:
                road_squid = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, z)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 0.1, 0.3))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(road_squid))
                glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Patrick
            model_patrick = glm.translate(glm.mat4(1.0), pos_patrick + glm.vec3(0.0,0.0,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,1.6,1.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_patrick))
            glBindVertexArray(sph_vao); glDrawArrays(GL_TRIANGLES,0,sph_count)
            
            # Флюгер Патрика
            weathervane_vertical = model_for_box((pos_patrick.x, pos_patrick.y + 1.5, pos_patrick.z), (0.1, 0.75, 0.1))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(weathervane_vertical))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            weathervane_horizontal = model_for_box((pos_patrick.x, pos_patrick.y + 1.92, pos_patrick.z), (0.6, 0.1, 0.1))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(weathervane_horizontal))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Squidward
            model_sq_base = glm.translate(glm.mat4(1.0), pos_squid) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,7.0,1.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_sq_base))
            glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)
            
            # Крыша Сквидварда (диск)
            roof_squid = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.0,3.5,0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.8,1.8,1.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(roof_squid))
            glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)
            
            # Детали Сквидварда
            ear_right = model_for_box((pos_squid.x + 1.0, pos_squid.y + 2.0, pos_squid.z), (0.2,0.8,0.2))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(ear_right))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            ear_left  = model_for_box((pos_squid.x - 1.0, pos_squid.y + 2.0, pos_squid.z), (0.2,0.8,0.2))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(ear_left))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            brow = model_for_box((pos_squid.x, pos_squid.y + 2.6, pos_squid.z + 0.8), (1.4,0.25,0.25))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(brow))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            nose = model_for_box((pos_squid.x, pos_squid.y + 2.0, pos_squid.z + 0.9), (0.4,1.0,0.4))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(nose))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Дверь Сквидварда
            door_squid = model_for_box((pos_squid.x, pos_squid.y + 0.5, pos_squid.z + 0.88), (0.6,1.2,0.05))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_squid))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # SpongeBob
            model_sponge = glm.translate(glm.mat4(1.0), pos_sponge) * glm.scale(glm.mat4(1.0), glm.vec3(1.5,2.8,1.5))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_sponge))
            glBindVertexArray(sph_vao); glDrawArrays(GL_TRIANGLES,0,sph_count)
            
            # Листья Спанчбоба
            leaf1 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, 3.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.8,0.6))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(leaf1))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            leaf2 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(-0.25, 2.9, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(45.0), glm.vec3(0,1,1)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.7,0.6))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(leaf2))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            leaf3 = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.25, 2.9, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(-45.0), glm.vec3(0,1,1)) * glm.scale(glm.mat4(1.0), glm.vec3(0.1,0.7,0.6))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(leaf3))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Дверь Спанчбоба
            door_sponge = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, -0.25, 1.45)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 1.3, 0.1))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_sponge))
            glBindVertexArray(sph_vao); glDrawArrays(GL_TRIANGLES,0,sph_count)
            
            # Рамка двери Спанчбоба
            door_frame_model = glm.translate(glm.mat4(1.0), pos_sponge + glm.vec3(0.0, 0.0, 1.45)) * glm.scale(glm.mat4(1.0), glm.vec3(1.0, 2.2, 0.8))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_frame_model))
            glBindVertexArray(window_frame_vao); glDrawArrays(GL_TRIANGLES,0,window_frame_count)
            
            # Красти Крабс - форма сундука
            # Основное здание (сундук)
            krusty_krab_base = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 2.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(6.0, 4.0, 5.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(krusty_krab_base))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            
            # Крышка сундука (полуцилиндр с закрытыми боками)
            krusty_krab_lid = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 4.0, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1.0, 0.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(6.0, 2.5, 5.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(krusty_krab_lid))
            glBindVertexArray(half_cyl_vao); glDrawArrays(GL_TRIANGLES,0,half_cyl_count)

            # Труба на крыше 
            krusty_krab_chimney = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(2.0, 5.0, -1.5)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6, 1.2, 0.6))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(krusty_krab_chimney))
            glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)

            # Большие прямоугольные окна 
            window1_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-1.8, 3.0, 2.51)) * glm.scale(glm.mat4(1.0), glm.vec3(1.2, 1.5, 0.5))
            window2_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(1.8, 3.0, 2.51)) * glm.scale(glm.mat4(1.0), glm.vec3(1.2, 1.5, 0.5))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window1_krab))
            glBindVertexArray(rectangular_window_vao); glDrawArrays(GL_TRIANGLES,0,rectangular_window_count)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window2_krab))
            glBindVertexArray(rectangular_window_vao); glDrawArrays(GL_TRIANGLES,0,rectangular_window_count)

            # Стеклянная дверь (ПЕРЕМЕЩЕНА НА ЗДАНИЕ)
            door_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 2.0, 2.51)) * glm.scale(glm.mat4(1.0), glm.vec3(1.5, 2.5, 0.1))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_krab))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)

            # Ручки двери
            door_handle_left = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-0.5, 1.0, 2.55)) * glm.scale(glm.mat4(1.0), glm.vec3(0.05, 0.2, 0.05))
            door_handle_right = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.5, 1.0, 2.55)) * glm.scale(glm.mat4(1.0), glm.vec3(0.05, 0.2, 0.05))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_handle_left))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(door_handle_right))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)

            # Основание
            foundation_front = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, -2.0, 2.6)) * glm.scale(glm.mat4(1.0), glm.vec3(7.2, 0.3, 0.3))
            foundation_back = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, -2.0, -2.6)) * glm.scale(glm.mat4(1.0), glm.vec3(7.2, 0.3, 0.3))
            foundation_left = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.6, -2.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.3, 0.3, 5.2))
            foundation_right = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(3.6, -2.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.3, 0.3, 5.2))

            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(foundation_front))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(foundation_back))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(foundation_left))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(foundation_right))
            glBindVertexArray(cube_vao); glDrawArrays(GL_TRIANGLES,0,cube_count)

            # Вывеска "Krusty Krab"
            sign_post_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.0, 0.0, 4.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.0, 1.0, 1.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(sign_post_model))
            glBindVertexArray(sign_post_vao); glDrawArrays(GL_TRIANGLES,0,sign_post_count)
            
            shell_sign_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.0, 2.5, 4.0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.2, 0.8, 1.0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(shell_sign_model))
            glBindVertexArray(shell_sign_vao); glDrawArrays(GL_TRIANGLES,0,shell_sign_count)
            
            # Гирлянда флажков с параболической веревкой
            rope_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 0.0, 4.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0))
            glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(rope_model))
            glBindVertexArray(rope_vao); glDrawArrays(GL_TRIANGLES,0,rope_count)
            
            # Флажки (прямоугольные) с поворотом по касательной к веревке
            flag_positions = [
                glm.vec3(-3.3, 3.35, 2.4),   # первый флаг
                glm.vec3(-2.1, 3.45, 2.4),   # второй  
                glm.vec3(-0.9, 3.55, 2.4),   # третий
                glm.vec3(0.3, 3.45, 2.4),    # четвертый
                glm.vec3(1.5, 3.35, 2.4),    # пятый
                glm.vec3(2.7, 3.25, 2.4)     # шестой
            ]
            
            for i, flag_pos in enumerate(flag_positions):
                flag_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_pos) * \
                        glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                        glm.rotate(glm.mat4(1.0), glm.radians(-25.0), glm.vec3(0,0,1))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(flag_model))
                glBindVertexArray(flag_vao); glDrawArrays(GL_TRIANGLES,0,flag_count)
        
            
            # Домики обычных жителей
            for i, house_pos in enumerate(house_positions):
                random_height_bonus = random_height_bonuses[i]

                # Тело домика
                model_house = glm.translate(glm.mat4(1.0), house_pos) * glm.scale(glm.mat4(1.0), glm.vec3(2.0, 10.0 + random_height_bonus, 2.0))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_house))
                glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)
                
                # Крыша домика
                model_roof_base = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.0 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 0.2, 2.2))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_roof_base))
                glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)
                
                # Диски крыши
                model_roof_bottom = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.0 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 1.0, 2.2))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_roof_bottom))
                glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)
                
                model_roof_top = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.1 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 1.0, 2.2))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_roof_top))
                glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)
                
                # Труба
                model_chimney = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(random_height_bonus / 4, 5.1 + random_height_bonus / 2, random_height_bonus / 4)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 2.0, 0.4))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_chimney))
                glBindVertexArray(cyl_vao); glDrawArrays(GL_TRIANGLES,0,cyl_count)
                
                # Диск трубы
                model_chimney_disk = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(random_height_bonus / 4, 5.9 + random_height_bonus / 2, random_height_bonus / 4)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 2.0, 0.4))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(model_chimney_disk))
                glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)
                
                # Окна (рамки)
                window_front = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 1.5 + random_height_bonus / 2, 1.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.75,0.75,0.75))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window_front))
                glBindVertexArray(window_frame_vao); glDrawArrays(GL_TRIANGLES,0,window_frame_count)
                
                window_back = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 4.0 + random_height_bonus / 2, -1.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.75,0.75,0.75))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window_back))
                glBindVertexArray(window_frame_vao); glDrawArrays(GL_TRIANGLES,0,window_frame_count)
                
                # Стекла окон
                window_glass_front = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 1.5 + random_height_bonus / 2, 1.0)) * glm.scale(glm.mat4(1.0) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)), glm.vec3(0.6, 0.6, 0.6))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window_glass_front))
                glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)
                
                window_glass_back = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 4.0 + random_height_bonus / 2, -1.0)) * glm.scale(glm.mat4(1.0) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)), glm.vec3(0.6, 0.6, 0.6))
                glUniformMatrix4fv(glGetUniformLocation(depth_prog,'model'),1,GL_FALSE, glm.value_ptr(window_glass_back))
                glBindVertexArray(disk_vao); glDrawArrays(GL_TRIANGLES,0,disk_count)

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

        # plane (sand)
        draw_textured(plane_vao, plane_count, tex_sand, glm.translate(glm.mat4(1.0), glm.vec3(0.0,-0.01,0.0)), 8.0)

        # Главная дорога
        road_main = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 8.0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
        draw_textured(cube_vao, cube_count, tex_road, road_main, 16.0)

        # Изогнутая дорога влево
        curved_road_model = glm.mat4(1.0)
        draw_textured(curved_road_vao, curved_road_count, tex_road, curved_road_model, 16.0)

        # Дорога дальше после изогнутой
        road_next = glm.translate(glm.mat4(1.0), glm.vec3(-45.0, 0.0, 53.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
        draw_textured(cube_vao, cube_count, tex_road, road_next, 16.0)

        # Дорога вправо
        road_to_right = glm.translate(glm.mat4(1.0), glm.vec3(40.0, 0.0, 8.0)) * glm.scale(glm.mat4(1.0), glm.vec3(40.0, 0.01, 3.0))
        draw_textured(cube_vao, cube_count, tex_road, road_to_right, 16.0)

        # Дорога до красти краба
        road_to_krusty_krabs = glm.translate(glm.mat4(1.0), glm.vec3(50.0, 0.0, 5.75)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.5, 0.01, 1.5))
        draw_textured(cube_vao, cube_count, tex_road, road_to_krusty_krabs, 16.0)

        # Асфальт под красти крабом
        under_krusty_krabs = glm.translate(glm.mat4(1.0), glm.vec3(50.0, 0.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(12.0, 0.01, 10.0))
        draw_textured(cube_vao, cube_count, tex_road, under_krusty_krabs, 16.0)

        # Красти Крабс - форма сундука
        # Основное здание (сундук)
        krusty_krab_base = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 0.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 5.0, 5.0))
        draw_textured(cube_vao, cube_count, tex_krusty_krab, krusty_krab_base, 32.0)

        # Крышка сундука (полуцилиндр с закрытыми боками)
        krusty_krab_lid = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 2.5, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0.0, 0.0, 1.0))  * glm.scale(glm.mat4(1.0), glm.vec3(5.0, 7.0, 5.0))
        draw_textured(half_cyl_vao, half_cyl_count, tex_krusty_krab, krusty_krab_lid, 32.0)

        # Труба на крыше 
        krusty_krab_chimney = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 5.0, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.6, 2.0, 0.6))
        draw_textured(cyl_vao, cyl_count, tex_metal, krusty_krab_chimney, 32.0)

        # Стекла окон
        window_glass1_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-2.0, 1.4, 2.52)) * glm.scale(glm.mat4(1.0), glm.vec3(1.9, 1.65, 0.1))
        window_glass2_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(2.0, 1.4, 2.52)) * glm.scale(glm.mat4(1.0), glm.vec3(1.9, 1.65, 0.1))
        draw_textured(cube_vao, cube_count, tex_window_blue, window_glass1_krab, 64.0)
        draw_textured(cube_vao, cube_count, tex_window_blue, window_glass2_krab, 64.0)

        # Стеклянная дверь
        door_krab = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 1.0, 2.51)) * glm.scale(glm.mat4(1.0), glm.vec3(1.5, 2.0, 0.1))
        draw_textured(cube_vao, cube_count, tex_glass_door, door_krab, 64.0)

        # 1) Ручки для стеклянной двери (золотистые)
        door_handle_left = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-0.1, 1.0, 2.55)) * glm.scale(glm.mat4(1.0), glm.vec3(0.05, 0.2, 0.05))
        door_handle_right = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.1, 1.0, 2.55)) * glm.scale(glm.mat4(1.0), glm.vec3(0.05, 0.2, 0.05))
        draw_textured(cube_vao, cube_count, tex_gold, door_handle_left, 64.0)
        draw_textured(cube_vao, cube_count, tex_gold, door_handle_right, 64.0)

        # 2) Деревянное основание внизу Красти Крабс
        foundation_front1 = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-2.3, 0.25, 2.6)) * glm.scale(glm.mat4(1.0), glm.vec3(2.9, 0.5, 0.3))
        foundation_front2 = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(2.3, 0.25, 2.6)) * glm.scale(glm.mat4(1.0), glm.vec3(2.9, 0.5, 0.3))
        foundation_back = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 0.25, -2.6)) * glm.scale(glm.mat4(1.0), glm.vec3(7.5, 0.5, 0.3))
        foundation_left = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.6, 0.25, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.3, 0.5, 4.9))
        foundation_right = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(3.6, 0.25, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.3, 0.5, 4.9))

        draw_textured(cube_vao, cube_count, tex_krusty_krab, foundation_front1, 24.0)
        draw_textured(cube_vao, cube_count, tex_krusty_krab, foundation_front2, 24.0)
        draw_textured(cube_vao, cube_count, tex_krusty_krab, foundation_back, 24.0)
        draw_textured(cube_vao, cube_count, tex_krusty_krab, foundation_left, 24.0)
        draw_textured(cube_vao, cube_count, tex_krusty_krab, foundation_right, 24.0)

        # ---- Вывеска "The Krusty Krab" ----
        # Столб вывески
        sign_post_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.5, 0.0, 5.28)) * glm.scale(glm.mat4(1.0), glm.vec3(1.2, 2.8, 1.0))
        draw_textured(sign_post_vao, sign_post_count, tex_wood, sign_post_model, 24.0)
        
        # Ракушка-вывеска
        shell_sign_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.5, 5.45, 5.57)) * glm.rotate(glm.mat4(1.0), glm.radians(-90.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(1.2, 1.0, 1.0)) 
        draw_textured(shell_sign_vao, shell_sign_count, tex_shell, shell_sign_model, 32.0)
        
        # Текст "The Krusty Krab" 
        sign_text_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(-3.5, 5.45, 5.58)) * glm.scale(glm.mat4(1.0), glm.vec3(2.5, 1.1, 0.01))
        draw_textured(cube_vao, cube_count, tex_sign_text, sign_text_model, 64.0)
        
        rope_model = glm.translate(glm.mat4(1.0), pos_krusty_krab + glm.vec3(0.0, 4.0, 2.0)) * glm.rotate(glm.mat4(1.0), glm.radians(145.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.7, 1.0, 1.0)) 
        draw_textured(rope_vao, rope_count, tex_rope, rope_model, 24.0)
        
        # Все флажки
        flag_positions = [
            glm.vec3(-3.3, 3.35, 2.4),  
            glm.vec3(-2.1, 2.87, 2.65),  
            glm.vec3(-0.9, 2.63, 2.78),  
            glm.vec3(0.4, 2.6, 2.8),   
            glm.vec3(1.6, 2.72, 2.72),   
            glm.vec3(2.65, 3.15, 2.57)     
        ]
        
        flag_model1 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[0]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-25.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[0], flag_model1, 32.0)
        flag_model2 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[1]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-17.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[1], flag_model2, 32.0)   
        flag_model3 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[2]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-5.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[2], flag_model3, 32.0)  
        flag_model4 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[3]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(5.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[3], flag_model4, 32.0) 
        flag_model5 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[4]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(17.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[4], flag_model5, 32.0) 
        flag_model6 = glm.translate(glm.mat4(1.0), pos_krusty_krab + flag_positions[5]) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(-30.0), glm.vec3(1,0,0)) * \
                       glm.rotate(glm.mat4(1.0), glm.radians(25.0), glm.vec3(0,0,1)) 
        draw_textured(flag_vao, flag_count, flag_colors[5], flag_model6, 32.0) 


        # Домики обычных жителей 
        for i, house_pos in enumerate(house_positions):
            random_height_bonus = random_height_bonuses[i]
            random_tex = random_texture_for_houses[i]

            # Тело домика (цилиндр)
            model_house = glm.translate(glm.mat4(1.0), house_pos) * glm.scale(glm.mat4(1.0), glm.vec3(2.0, 10.0 + random_height_bonus, 2.0))
            draw_textured(cyl_vao, cyl_count, random_tex, model_house, 24.0)
            
            # Крыша домика
            # Основание крыши (цилиндр с небольшой высотой)
            model_roof_base = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.0 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 0.2, 2.2))
            draw_textured(cyl_vao, cyl_count, random_tex, model_roof_base, 24.0)

            # нижняя часть крыши
            model_roof_bottom = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.0 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 1.0, 2.2))
            draw_textured(disk_vao, disk_count, random_tex, model_roof_bottom, 24.0)
            
            # Верхняя часть крыши (диск)
            model_roof_top = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 5.1 + random_height_bonus / 2, 0.0)) * glm.scale(glm.mat4(1.0), glm.vec3(2.2, 1.0, 2.2))
            draw_textured(disk_vao, disk_count, random_tex, model_roof_top, 24.0)

            # Труба на крыше
            model_chimney = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(random_height_bonus / 4, 5.1 + random_height_bonus / 2, random_height_bonus / 4)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 2.0, 0.4))
            draw_textured(cyl_vao, cyl_count, random_tex, model_chimney, 24.0)
            model_chimney_disk = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(random_height_bonus / 4, 5.9 + random_height_bonus / 2, random_height_bonus / 4)) * glm.scale(glm.mat4(1.0), glm.vec3(0.4, 2.0, 0.4))
            draw_textured(disk_vao, disk_count, random_tex, model_chimney_disk, 24.0)
            
            # Окна домика
            window_front = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 1.5 + random_height_bonus / 2, 1.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.75,0.75,0.75))
            window_back = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 4.0 + random_height_bonus / 2, -1.0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.75,0.75,0.75))
          
            # Рамки окон
            draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_front, 64.0)
            draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_back, 64.0)
            
            # Стекла окон
            window_glass_front = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 1.5 + random_height_bonus / 2, 1.0)) * glm.scale(glm.mat4(1.0) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)), glm.vec3(0.6, 0.6, 0.6))
            window_glass_back = glm.translate(glm.mat4(1.0), house_pos + glm.vec3(0.0, 4.0 + random_height_bonus / 2, -1.0)) * glm.scale(glm.mat4(1.0) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)), glm.vec3(0.6, 0.6, 0.6))
      
            draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_front, 64.0)
            draw_textured(disk_vao, disk_count, tex_window_blue, window_glass_back, 64.0)

        # Дорожка к дому Патрика
        road_patrick = glm.translate(glm.mat4(1.0), glm.vec3(-8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.01, 1.0))
        draw_textured(cube_vao, cube_count, tex_road, road_patrick, 16.0)
        
        # Дорожка к дому Спанчбоба
        road_sponge = glm.translate(glm.mat4(1.0), glm.vec3(8.0, 0.0, 3.0)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0)) * glm.scale(glm.mat4(1.0), glm.vec3(7.0, 0.01, 1.0))
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
        window_left = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(-0.45, 2.2, 0.86)) * glm.scale(glm.mat4(1.0), glm.vec3(0.5,0.5,0.5))
        window_right = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.45, 2.2, 0.86)) * glm.scale(glm.mat4(1.0), glm.vec3(0.5,0.5,0.5))
        
        # Рамки окон (объемные)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_left, 64.0)
        draw_textured(window_frame_vao, window_frame_count, tex_window_frame, window_right, 64.0)
        
        # Стекла окон (немного смещены назад относительно рамок)
        window_glass_left = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(-0.45, 2.2, 0.86)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.35,0.35,0.35))
        window_glass_right = glm.translate(glm.mat4(1.0), pos_squid + glm.vec3(0.45, 2.2, 0.86)) * glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(1,0,0)) * glm.scale(glm.mat4(1.0), glm.vec3(0.35,0.35,0.35))
        
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
        
        # Рендер солнца (после всех непрозрачных объектов, но до полупрозрачных)
        glUseProgram(sun_prog)
        glUniformMatrix4fv(glGetUniformLocation(sun_prog, 'view'), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(sun_prog, 'projection'), 1, GL_FALSE, glm.value_ptr(proj))
        glUniform3fv(glGetUniformLocation(sun_prog, 'sunPos'), 1, glm.value_ptr(light_pos))
        glUniform3fv(glGetUniformLocation(sun_prog, 'viewPos'), 1, glm.value_ptr(cam_pos))
        
        # Отключаем тест глубины для солнца, чтобы оно всегда было видно
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        
        # Создаем солнце как сферу с ярким свечением
        sun_model = glm.translate(glm.mat4(1.0), light_pos) * glm.scale(glm.mat4(1.0), glm.vec3(8.0, 8.0, 8.0))
        glUniformMatrix4fv(glGetUniformLocation(sun_prog, 'model'), 1, GL_FALSE, glm.value_ptr(sun_model))
        glBindVertexArray(sph_vao)
        glDrawArrays(GL_TRIANGLES, 0, sph_count)
        
        # Включаем тест глубины обратно
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        
        # Рендер цветков (после всей сцены, чтобы прозрачность работала правильно)
        glUseProgram(flower_prog)
        glUniformMatrix4fv(glGetUniformLocation(flower_prog, 'view'), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(flower_prog, 'projection'), 1, GL_FALSE, glm.value_ptr(proj))
        
        glBindVertexArray(flower_vao)
        
        for pos, flower_type in flower_positions:
            # Привязываем текстуру в зависимости от типа цветка
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex_flowers[flower_type])
            glUniform1i(glGetUniformLocation(flower_prog, 'texture_diffuse1'), 0)
            
            # Создаем билборд - цветок всегда повернут к камере
            model = glm.translate(glm.mat4(1.0), pos)
            
            # Вычисляем направление от цветка к камере
            to_camera = glm.normalize(cam_pos - pos)
            # Создаем матрицу поворота чтобы цветок смотрел на камеру
            up = glm.vec3(0.0, 1.0, 0.0)
            right = glm.normalize(glm.cross(up, to_camera))
            up = glm.normalize(glm.cross(to_camera, right))
            
            # Применяем поворот
            model[0][0] = right.x; model[0][1] = right.y; model[0][2] = right.z
            model[1][0] = up.x;    model[1][1] = up.y;    model[1][2] = up.z
            model[2][0] = to_camera.x; model[2][1] = to_camera.y; model[2][2] = to_camera.z
            
            glUniformMatrix4fv(glGetUniformLocation(flower_prog, 'model'), 1, GL_FALSE, glm.value_ptr(model))
            glDrawArrays(GL_TRIANGLES, 0, flower_count)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == '__main__':
    main()