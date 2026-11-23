"""3D anatomy from the Human Reference Atlas (HRA), with GTEx's registered sampling sites.

Organs are the HRA 3D reference organs (Visible Human male and female, CC BY 4.0),
placed as the HRA Exploration User Interface places them (the CCF API scene). Sampling
sites are the tissue blocks that the GTEx project registered into those organs with
the HRA Registration User Interface (hra-registrations, gtex). Where GTEx registered
no block, the site is the named HRA sub-structure, or, for organs the HRA does not
model (stomach, oesophagus, adrenal, thyroid gland, testis, pituitary, leg muscle,
nerve and artery), a point derived from neighbouring structures and drawn as
approximate.

Coordinates are metres: +x the subject's left, +y up, +z anterior."""
import json
import re
from functools import lru_cache

import numpy as np

from .inputs import ASSETS, ROOT

HRA = ASSETS / 'hra'            # pinned organ meshes and the rendered-card cache
SCENES = ROOT / 'data' / 'hra'  # CCF API scenes, kept in the repository because the API is unversioned


@lru_cache(maxsize=1)
def scene_digest():
    import hashlib
    h = hashlib.sha256()
    for sex in ('female', 'male'):
        h.update((SCENES / f'scene-{sex}.json').read_bytes())
    return h.hexdigest()[:16]

# GTEx tissue (as coded in IDC) -> organ label in the HRA scene, optional sub-structure
# pattern, GTEx registration name, preferred body, and how to place an estimated site.
TISSUES = {
    'Lung': dict(organ='lung', block='Lung'),
    'Heart': dict(organ='heart', sub='left_ventricle', block='Heart_Left_Ventricle'),
    'Auricular appendage': dict(organ='heart', sub='right_cardiac_atrium', block='Heart_Atrial_Appendage'),
    'Coronary artery': dict(organ='blood vasculature', sub='coronary_artery', block='Artery_Coronary'),
    'Aorta': dict(organ='blood vasculature', sub='descending_aorta_a'),
    'Liver': dict(organ='liver', block='Liver'),
    'Spleen': dict(organ='spleen', block='Spleen'),
    'Pancreas': dict(organ='pancreas', block='Pancreas'),
    'Renal cortex': dict(organ='Left kidney', block='Kidney_Cortex'),
    'Renal medulla': dict(organ='Left kidney'),
    # GTEx's two colon registrations carry swapped names: the block registered as
    # Colon_Sigmoid lies in, and is annotated as, the transverse colon (UBERON:0001157),
    # and vice versa. Colon blocks are therefore chosen by their anatomical annotation.
    'Transverse colon': dict(organ='large intestine', sub='transverse_colon', uberon='UBERON_0001157'),
    'Sigmoid colon': dict(organ='large intestine', sub='sigmoid_colon', uberon='UBERON_0001159', sex='male'),
    'Ileum': dict(organ='small intestine', sub='ileum_terminal', block='Small_Intestine_Terminal_Ileum'),
    'Omentum': dict(organ='omentum'),
    'Urinary bladder': dict(organ='urinary bladder'),
    'Prostate': dict(organ='prostate', block='Prostate', sex='male'),
    'Uterus': dict(organ='uterus', block='Uterus', sex='female'),
    'Ovary': dict(organ='Left ovary', block='Ovary', sex='female'),
    'Fallopian tube': dict(organ='Left fallopian tube', sex='female'),
    'Endocervix': dict(organ='uterus', sub='internal_cervical_os', sex='female'),
    'Exocervix': dict(organ='uterus', sub='external_cervical_os', sex='female'),
    'Vagina': dict(organ='uterus', sub='cervicovaginal_junction', sex='female', approx=True),
    'Breast': dict(organ='Right mammary gland', block='Breast', sex='female'),
    'Skin of suprapubic region': dict(organ='skin', block='Skin_Not_Sun_Exposed_Suprapubic'),
    'Skin of lower leg': dict(organ='skin', block='Skin_Sun_Exposed_Lower_leg'),
    # Lower-leg tissues: GTEx takes them beside the lower-leg skin sample. The HRA has
    # no leg muscle, nerve or artery, so they sit at that block, deeper, as approximate.
    'Adipose tissue': dict(block='Skin_Sun_Exposed_Lower_leg', depth=.008, approx=True),
    'Skeletal muscle': dict(block='Skin_Sun_Exposed_Lower_leg', depth=.03, approx=True),
    'Tibial artery': dict(block='Skin_Sun_Exposed_Lower_leg', depth=.04, approx=True),
    'Tibial nerve': dict(block='Skin_Sun_Exposed_Lower_leg', depth=.045, approx=True),
    'Cerebral cortex': dict(organ='brain', sub='frontal'),
    'Cerebellum': dict(organ='brain', sub='cerebell'),
    'Pituitary': dict(organ='brain', derive='pituitary', approx=True),
    'Minor salivary gland': dict(organ='mouth', sub='frenulum_of_lower_lip|buccal_mucosa'),
    'Thyroid': dict(organ='larynx', sub='thyroid_cartilage', derive='thyroid', approx=True),
    'Esophageal mucous membrane': dict(derive='oesophagus', approx=True),
    'Esophageal muscle': dict(derive='oesophagus', approx=True),
    'Gastroesophageal junction': dict(derive='gej', approx=True),
    'Stomach': dict(derive='stomach', approx=True),
    'Adrenal gland': dict(organ='Left kidney', derive='adrenal', approx=True),
    'Testis': dict(derive='testis', sex='male', approx=True),
}

# Context organs drawn in every view, dim.
CONTEXT = ('brain', 'lung', 'heart', 'liver', 'spleen', 'pancreas', 'Left kidney', 'Right kidney',
           'large intestine', 'small intestine', 'urinary bladder', 'uterus', 'prostate', 'trachea',
           'blood vasculature', 'pelvis', 'spinal cord', 'Left mammary gland', 'Right mammary gland')


def matrix(values):
    """deck.gl / CCF scene matrices are column-major."""
    return np.asarray(values, np.float64).reshape(4, 4).T


class Mesh:
    def __init__(self, name, vertices, faces, normals):
        self.name, self.v, self.f, self.n = name, vertices, faces, normals


@lru_cache(maxsize=2)
def body(sex):
    return Body(sex)


class Body:
    def __init__(self, sex):
        import trimesh
        self.sex = sex
        scene = json.loads((SCENES / f'scene-{sex}.json').read_text())
        self.organs = {}
        self.blocks = {}
        self.block_terms = {}
        for node in scene:
            if not node.get('transformMatrix'):
                continue
            if node.get('scenegraph'):
                path = HRA / node['scenegraph'].split('/')[-1]
                if not path.exists() or node['tooltip'] in self.organs:
                    continue
                M = matrix(node['transformMatrix'])
                meshes = []
                for name, g in self._geometries(trimesh.load(path, force='scene')):
                    with np.errstate(all='ignore'):   # Accelerate raises spurious flags here
                        v = (np.c_[g.vertices, np.ones(len(g.vertices))] @ M.T)[:, :3]
                        n = g.vertex_normals @ M[:3, :3].T
                    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                    meshes.append(Mesh(name, v.astype(np.float32), g.faces.astype(np.int32), n.astype(np.float32)))
                self.organs[node['tooltip']] = meshes
            elif node.get('entityId', '').startswith('https://gtexportal.org'):
                tissue = node['entityId'].split('/')[-1].split('#')[0]
                self.blocks[tissue] = matrix(node['transformMatrix'])
                for term in node.get('ccf_annotations', []):
                    self.block_terms.setdefault(term.split('/')[-1], tissue)

    @staticmethod
    def _geometries(scene):
        for node in scene.graph.nodes_geometry:
            transform, key = scene.graph[node]
            g = scene.geometry[key].copy()
            g.apply_transform(transform)
            yield node, g

    def parts(self, organ, pattern=None):
        meshes = self.organs.get(organ, [])
        if pattern:
            meshes = [m for m in meshes if re.search(pattern, m.name, re.I)]
        return meshes

    def points(self, organ, pattern=None):
        parts = self.parts(organ, pattern)
        return np.concatenate([m.v for m in parts]) if parts else np.empty((0, 3), np.float32)

    def block_corners(self, name):
        M = self.blocks[name]
        cube = np.array([[x, y, z, 1] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        return (cube @ M.T)[:, :3]

    def site(self, tissue):
        """(point, block corners or None, approximate?) for a GTEx tissue."""
        spec = TISSUES.get(tissue, {})
        corners = None
        name = self.block_terms.get(spec.get('uberon'), spec.get('block'))
        if name in self.blocks:
            spec = dict(spec, block=name)
            corners = self.block_corners(name)
            point = corners.mean(axis=0)
            if spec.get('depth'):
                # Move inward from the skin, toward the limb's axis at that height.
                skin = self.points('skin')
                near = skin[np.abs(skin[:, 1] - point[1]) < .01]
                side = near[np.sign(near[:, 0]) == np.sign(point[0])]
                axis = side.mean(axis=0) if len(side) else point
                d = axis - point
                d[1] = 0
                point = point + d/np.linalg.norm(d)*spec['depth']
                corners = None
            return point, corners, bool(spec.get('approx'))
        derive = spec.get('derive')
        if derive:
            return self._derive(derive), None, True
        pts = self.points(spec.get('organ', ''), spec.get('sub'))
        if len(pts):
            return pts.mean(axis=0), None, bool(spec.get('approx'))
        return None, None, True

    def _derive(self, what):
        def c(organ, pattern=None):
            p = self.points(organ, pattern)
            return p.mean(axis=0) if len(p) else None
        if what == 'pituitary':
            b = self.points('brain')
            lo = b[:, 1].min()
            return np.array([b[:, 0].mean(), lo + .012, b[:, 2].mean() + .012])
        if what == 'thyroid':
            t = self.points('trachea')
            top = t[t[:, 1] > t[:, 1].max() - .03]
            return np.array([top[:, 0].mean(), top[:, 1].mean() - .005, top[:, 2].max() + .006])
        if what == 'oesophagus':
            t = self.points('trachea')
            mid = t[np.abs(t[:, 1] - np.median(t[:, 1])) < .02]
            return np.array([mid[:, 0].mean() - .005, mid[:, 1].mean(), mid[:, 2].min() - .012])
        if what in ('gej', 'stomach'):
            liver, spleen, heart = c('liver'), c('spleen'), c('heart')
            p = (liver + 2*spleen)/3 if what == 'stomach' else (liver + spleen)/2
            if what == 'gej':
                p[1] = heart[1] - .09
            return p + np.array([0, 0, .03])
        if what == 'adrenal':
            k = self.points('Left kidney')
            top = k[k[:, 1] > k[:, 1].max() - .015]
            return top.mean(axis=0) + np.array([-.01, .015, 0])
        if what == 'testis':
            p = self.points('pelvis')
            return np.array([p[:, 0].mean(), p[:, 1].min() - .05, p[:, 2].max() - .01])
        raise KeyError(what)


def sex_for(tissue, available=('female', 'male')):
    return TISSUES.get(tissue, {}).get('sex', 'female')


# ---------------------------------------------------------------- rendering
VERT = '''
#version 330
uniform mat4 mvp;
uniform mat3 rot;
in vec3 pos;
in vec3 nrm;
out vec3 n_view;
void main() {
    n_view = rot * nrm;
    gl_Position = mvp * vec4(pos, 1.0);
}
'''
FRAG = '''
#version 330
uniform vec4 colour;
uniform int mode;
in vec3 n_view;
out vec4 out_colour;
void main() {
    vec3 n = normalize(n_view);
    if (!gl_FrontFacing) n = -n;
    float rim = 1.0 - abs(n.z);
    if (mode == 1) {           // skin: a glass shell lit at its silhouette
        float a = colour.a * (0.035 + 0.965 * pow(rim, 3.0));
        out_colour = vec4(colour.rgb, a);
        return;
    }
    vec3 L = normalize(vec3(-0.35, 0.55, 0.75));
    float diff = max(dot(n, L), 0.0);
    float light = 0.38 + 0.62 * diff;
    vec3 rgb = colour.rgb * light + vec3(0.10) * pow(rim, 2.0);
    if (mode == 2) rgb = colour.rgb * (0.75 + 0.25 * diff) + vec3(0.25) * pow(rim, 1.5);
    out_colour = vec4(rgb, colour.a);
}
'''


class Renderer:
    def __init__(self):
        import moderngl
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vaos = {}

    def vao(self, mesh):
        # Organ meshes live for the whole process and are uploaded once; transient
        # meshes (tissue-block cubes) are uploaded per render and released after.
        key = id(mesh)
        if key in self.vaos and self.vaos[key][0] is mesh:
            return self.vaos[key][1]
        data = np.hstack([mesh.v, mesh.n]).astype('f4')
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(mesh.f.astype('i4').tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f', 'pos', 'nrm')], ibo)
        if mesh.name != 'block':
            self.vaos[key] = (mesh, vao)
        return vao

    def render(self, b, size, centre, half_height, yaw_deg, items, background, skin_colour, skin_alpha=.7,
               ss=2, pitch_deg=0.):
        """items: list of (meshes, rgba 0-1, mode, depth_test). Returns RGB uint8 and a
        projector from body coordinates to output pixels."""
        import moderngl
        W, H = size[0]*ss, size[1]*ss
        fbo = self.ctx.simple_framebuffer((W, H), samples=4)
        out = self.ctx.simple_framebuffer((W, H))
        fbo.use()
        fbo.clear(*[c/255 for c in background], 1.0)
        yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R = Rx @ Ry
        aspect = W / H
        hw = half_height * aspect
        S = np.diag([1/hw, 1/half_height, -1/2.5])
        model = np.eye(4)
        model[:3, :3] = S @ R
        model[:3, 3] = -(S @ R @ np.asarray(centre, float))
        mvp = model.astype('f4')
        self.prog['mvp'].write(mvp.T.tobytes())
        self.prog['rot'].write(R.astype('f4').T.tobytes())
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for meshes, rgba, mode, depth in items:
            if depth:
                self.ctx.enable(moderngl.DEPTH_TEST)
            else:
                self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog['colour'].value = tuple(rgba)
            self.prog['mode'].value = mode
            for m in meshes:
                self.vao(m).render()
        # Skin last, depth-tested so organs show through the shell, without writing depth.
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.prog['colour'].value = (*[c/255 for c in skin_colour], skin_alpha)
        self.prog['mode'].value = 1
        for m in b.parts('skin'):
            self.vao(m).render()
        self.ctx.depth_mask = True
        self.ctx.copy_framebuffer(out, fbo)
        img = np.frombuffer(out.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
        fbo.release()
        out.release()
        import cv2
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        def project(p):
            q = model[:3, :3] @ np.asarray(p, float) + model[:3, 3]
            return ((q[0]+1)/2*size[0], (1-q[1])/2*size[1])
        return img, project


@lru_cache(maxsize=1)
def renderer():
    return Renderer()


def mix(a, b, t):
    return tuple((1-t)*x + t*y for x, y in zip(a, b))


CUBE_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                       [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)


def cube_mesh(corners):
    c = np.asarray(corners, np.float32)
    # Flat-shaded: one vertex per face corner.
    v = c[CUBE_FACES.ravel()]
    tri = v.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    centre = c.mean(axis=0)
    flip = np.sign(((tri.mean(axis=1) - centre)*n).sum(axis=1))[:, None]
    n = np.repeat(n*flip, 3, axis=0)
    return Mesh('block', v, np.arange(len(v), dtype=np.int32).reshape(-1, 3), n.astype(np.float32))


def marker_cube(point, size=.006):
    d = size/2
    return cube_mesh([point + np.array([x, y, z])*d for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def view(tissue, size, theme, zoom='body', yaw=0., sex=None, pitch=0.):
    """Render the body (zoom='body') or the sampled organ (zoom='organ') with the
    tissue's organ lit and its GTEx sampling site as a block. Returns the RGB image,
    the site (pixels, approximate flag, block outline) and the body's sex."""
    sex = sex or sex_for(tissue)
    b = body(sex)
    spec = TISSUES.get(tissue, {})
    point, corners, approx = b.site(tissue)
    bg = theme['map_bg']
    organ_rgb = [c/255 for c in theme['map_organ']]
    accent = [c/255 for c in theme['accent_rgb']]
    block_rgb = [c/255 for c in theme.get('block_rgb', (255, 255, 255))]
    lit_organ = spec.get('organ') if spec.get('organ') not in (None, 'skin') else None
    lit = b.parts(lit_organ, spec.get('sub')) if lit_organ else []
    lit_ids = {id(m) for m in lit}
    rest_of_organ = [m for m in b.parts(lit_organ) if id(m) not in lit_ids] if lit_organ else []
    block = cube_mesh(corners) if corners is not None else (marker_cube(point) if point is not None else None)
    items = []
    if zoom == 'organ' and lit_organ and not approx:
        # The organ on its own, like a gross specimen, with the sampled block.
        items.append((rest_of_organ, (*mix(organ_rgb, (1, 1, 1), .15), .55 if lit else .9), 0, True))
        if lit:
            items.append((lit, (*(accent if spec.get('sub') else mix(organ_rgb, accent, .6)), .9), 2, True))
        pts = b.points(lit_organ)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = (lo + hi)/2
        R = np.array([[np.cos(np.deg2rad(yaw)), 0, np.sin(np.deg2rad(yaw))], [0, 1, 0], [-np.sin(np.deg2rad(yaw)), 0, np.cos(np.deg2rad(yaw))]])
        q = (pts - centre) @ R.T
        half = .56*max(np.ptp(q[:, 1]), np.ptp(q[:, 0])*size[1]/size[0])
        skin_alpha = 0.
    else:
        for name in CONTEXT:
            meshes = [m for m in b.parts(name) if id(m) not in lit_ids]
            if meshes:
                items.append((meshes, (*organ_rgb, .8), 0, True))
        if lit:
            # Drawn over the other organs so posterior organs such as the kidney show.
            items.append((lit, (*(accent if spec.get('sub') else mix(organ_rgb, accent, .75)), .92), 2, False))
        if zoom == 'organ':
            centre = point if point is not None else b.points('skin').mean(axis=0)
            half = .16
        else:
            skin = b.points('skin')
            lo, hi = skin.min(axis=0), skin.max(axis=0)
            centre = (lo + hi)/2
            half = max(.52*(hi[1]-lo[1]), .52*(hi[0]-lo[0])*size[1]/size[0])
        skin_alpha = .85 if zoom == 'body' else .7
    if block is not None:
        items.append(([block], (*block_rgb, 1.), 2, False))
    img, project = renderer().render(b, size, centre, half, yaw, items, bg, theme['map_line'],
                                     skin_alpha=skin_alpha, pitch_deg=pitch)
    site = None
    if point is not None:
        xy = project(point)
        site = dict(xy=(float(xy[0]), float(xy[1])), approx=bool(approx),
                    corners=[tuple(float(v) for v in project(c)) for c in corners] if corners is not None else None)
    return img, site, sex
