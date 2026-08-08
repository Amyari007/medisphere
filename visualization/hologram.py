"""
Renders a 3D GLB mesh (e.g. the brain hologram) with a real OpenGL
pipeline — a genuinely different rendering path from the 2D OpenCV
drawing used everywhere else in this app — and composites the result
onto the webcam frame.

This needs a working OpenGL context (via GLFW) and a graphics driver on
the machine it runs on. It was built and tested in a Linux sandbox using
a software renderer (Mesa llvmpipe under Xvfb); behavior on your actual
Windows GPU driver is the next real test. If context creation or mesh
loading fails for any reason, this fails gracefully and the app falls
back to the procedural NeuroSphere instead of crashing — see
HologramRenderer.available.
"""

import ctypes
import os

import numpy as np

try:
    import glfw
    from OpenGL.GL import *
    import trimesh
    _DEPS_AVAILABLE = True
except Exception:
    _DEPS_AVAILABLE = False

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_world_pos = (u_model * vec4(in_position, 1.0)).xyz;
}
"""

# Holographic look: rim-lit cyan wireframe-ish shading using a fresnel
# term (view-dependent glow at silhouette edges), additive-friendly.
FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;

uniform vec3 u_view_pos;
uniform vec3 u_base_color;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_view_pos - v_world_pos);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.2);
    float core = 0.65;
    float intensity = core + fresnel * 1.3;
    vec3 color = u_base_color * intensity;
    float alpha = clamp(core + fresnel * 0.9, 0.0, 1.0);
    frag_color = vec4(color, alpha);
}
"""


def _compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader


def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up):
    eye, target, up = np.array(eye, dtype=np.float32), np.array(target, dtype=np.float32), np.array(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


def _rotation_y(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


class HologramRenderer:
    """
    Offscreen-rendered 3D hologram, composited onto a 2D frame each call.
    Construct once, call render_overlay(width, height, rotation_deg, scale)
    each frame to get an RGBA numpy array to blend onto the webcam frame.
    """

    def __init__(self, glb_path, render_w=640, render_h=480):
        self.available = False
        self._glb_path = glb_path
        self._render_w = render_w
        self._render_h = render_h
        self._window = None
        self._fbo = None
        self._program = None
        self._vao = None
        self._n_indices = 0

        if not _DEPS_AVAILABLE:
            print("[hologram] PyOpenGL/glfw/trimesh not available - falling back to procedural sphere.")
            return
        if not os.path.exists(glb_path):
            print(f"[hologram] model not found at {glb_path} - falling back to procedural sphere.")
            return

        try:
            self._init_gl_context()
            self._load_mesh(glb_path)
            self._init_shaders()
            self._init_fbo()
            self.available = True
        except Exception as e:
            print(f"[hologram] setup failed ({e}) - falling back to procedural sphere.")
            self.available = False

    def _init_gl_context(self):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed - no display/driver available")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(self._render_w, self._render_h, "hologram", None, None)
        if not self._window:
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self._window)

    def _load_mesh(self, glb_path):
        scene_or_mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # normalize to fit in a unit-ish sphere centered at origin, so any
        # source model (any real-world scale) renders consistently
        mesh.vertices -= mesh.bounding_box.centroid
        scale = 1.0 / max(mesh.bounding_box.extents.max(), 1e-6)
        mesh.vertices *= scale

        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.fix_normals()

        vertices = mesh.vertices.astype(np.float32)
        normals = mesh.vertex_normals.astype(np.float32)
        indices = mesh.faces.astype(np.uint32)
        self._n_indices = indices.size

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        vbo_norm = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_norm)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._vao = vao

    def _init_shaders(self):
        vs = _compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = _compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())
        glDeleteShader(vs)
        glDeleteShader(fs)
        self._program = program

    def _init_fbo(self):
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self._render_w, self._render_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)

        depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self._render_w, self._render_h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("framebuffer incomplete")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._fbo = fbo
        self._color_tex = color_tex

    def render_overlay(self, rotation_deg=0.0, scale=1.0, base_color=(0.3, 0.85, 1.0)):
        """
        Renders the mesh offscreen and returns an (H, W, 4) uint8 RGBA
        array ready to alpha-composite onto a video frame. Returns None
        if the renderer isn't available (caller should fall back).
        """
        if not self.available:
            return None

        glfw.make_context_current(self._window)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self._render_w, self._render_h)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        model = _rotation_y(rotation_deg)
        scale_m = np.eye(4, dtype=np.float32) * scale
        scale_m[3, 3] = 1.0
        model = model @ scale_m

        eye = (0, 0, 3.2)
        view = _look_at(eye, (0, 0, 0), (0, 1, 0))
        proj = _perspective(45.0, self._render_w / self._render_h, 0.1, 100.0)
        mvp = proj @ view @ model

        glUseProgram(self._program)
        glUniformMatrix4fv(glGetUniformLocation(self._program, "u_mvp"), 1, GL_TRUE, mvp)
        glUniformMatrix4fv(glGetUniformLocation(self._program, "u_model"), 1, GL_TRUE, model)
        glUniform3f(glGetUniformLocation(self._program, "u_view_pos"), *eye)
        glUniform3f(glGetUniformLocation(self._program, "u_base_color"), *base_color)

        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._n_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        pixels = glReadPixels(0, 0, self._render_w, self._render_h, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(self._render_h, self._render_w, 4)
        arr = np.flipud(arr)  # OpenGL reads bottom-up
        return arr

    @property
    def render_width(self):
        return self._render_w

    def close(self):
        if self._window is not None:
            try:
                glfw.destroy_window(self._window)
            except Exception:
                pass


def composite_rgba_onto_bgr(frame_bgr, overlay_rgba, center_xy, overlay_scale=1.0):
    """Alpha-blends an RGBA render onto a BGR OpenCV frame at center_xy."""
    if overlay_rgba is None:
        return
    oh, ow = overlay_rgba.shape[:2]
    if overlay_scale != 1.0:
        import cv2
        new_w, new_h = max(1, int(ow * overlay_scale)), max(1, int(oh * overlay_scale))
        overlay_rgba = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        oh, ow = overlay_rgba.shape[:2]

    cx, cy = center_xy
    x0, y0 = cx - ow // 2, cy - oh // 2
    x1, y1 = x0 + ow, y0 + oh

    fh, fw = frame_bgr.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(fw, x1), min(fh, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)

    region = overlay_rgba[sy0:sy1, sx0:sx1]
    alpha = (region[:, :, 3:4].astype(np.float32) / 255.0)
    rgb = region[:, :, [2, 1, 0]].astype(np.float32)  # RGBA->BGR channel order

    dest = frame_bgr[dy0:dy1, dx0:dx1].astype(np.float32)
    blended = rgb * alpha + dest * (1 - alpha)
    frame_bgr[dy0:dy1, dx0:dx1] = blended.astype(np.uint8)
