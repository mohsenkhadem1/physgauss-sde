"""Blender (Cycles) renders of a Figure-1 scene exported by blender_export.py. One image sequence per panel:
panel 0 = reference (true deformed surface), panels 1..5 = methods (their own particles, coloured by motion error,
over a faint ghost of the true surface; the method's own striker is solid, the true striker a ghost).
usage: blender -b -P blender_render.py -- SCENE.npz OUTDIR [--panels 0 1 ..] [--frames a b] [--samples 64] [--res 960 600]"""
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--')+1:]
SRC, OUTDIR = Path(argv[0]), Path(argv[1])
opt = lambda k, n, d: [type(x)(v) for x, v in zip(d, argv[argv.index(k)+1:argv.index(k)+1+n])] if k in argv else d
SAMPLES = opt('--samples', 1, [64])[0]; RES = opt('--res', 2, [960, 600])
Z = np.load(SRC, allow_pickle=False)
NF = int(Z['n'])
if '--panels' in argv:
    i = argv.index('--panels')+1; PANELS = []
    while i < len(argv) and not argv[i].startswith('--'):
        PANELS.append(int(argv[i])); i += 1
else:
    PANELS = list(range(6))
FR = opt('--frames', 2, [0, NF]); FRAMES = range(FR[0], FR[1])
SURFACE = '--points' not in argv
SCEN = SRC.stem.split('-')[0]
VIOLET = ['#d8d3f3', '#9085e9', '#6250d6', '#3a2c9c', '#1d1456']


def hex2rgb(h):
    h = h.lstrip('#'); c = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return [x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4 for x in c]+[1.0]  # sRGB -> linear


# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == 'OPTIX'
sc.cycles.device = 'GPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True; sc.cycles.denoiser = 'OPTIX'
sc.render.resolution_x, sc.render.resolution_y = RES; sc.render.film_transparent = False
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
sc.render.image_settings.file_format = 'PNG'

world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']; bg.inputs[0].default_value = hex2rgb('#eef0f3'); bg.inputs[1].default_value = 0.35


def mat_principled(name, color, rough=.5, metal=0., alpha=1., sss=0.):
    m = bpy.data.materials.new(name); m.use_nodes = True; p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = hex2rgb(color); p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal; p.inputs['Alpha'].default_value = alpha
    if sss:
        p.inputs['Subsurface Weight'].default_value = sss; p.inputs['Subsurface Radius'].default_value = (0.01, 0.01, 0.01)
    if alpha < 1:
        m.blend_method = 'BLEND'
    return m


def mat_error(vmax):
    m = bpy.data.materials.new('err'); m.use_nodes = True; nt = m.node_tree; p = nt.nodes['Principled BSDF']
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = 'err'; a.attribute_type = 'GEOMETRY'
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Max'].default_value = vmax
    cr = nt.nodes.new('ShaderNodeValToRGB'); el = cr.color_ramp.elements
    for i, h in enumerate(VIOLET):
        e = el[i] if i < 2 else el.new(i/4); e.position = i/4; e.color = hex2rgb(h)
    nt.links.new(a.outputs['Fac'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], cr.inputs['Fac'])
    nt.links.new(cr.outputs['Color'], p.inputs['Base Color']); p.inputs['Roughness'].default_value = .45
    return m


def add_box(name, lo, hi, mat):
    c = T(np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])]))
    lo, hi = c.min(0), c.max(0)
    bpy.ops.mesh.primitive_cube_add(location=tuple((lo+hi)/2)); o = bpy.context.object; o.name = name
    o.scale = tuple((hi-lo)/2); o.data.materials.append(mat)
    bv = o.modifiers.new('bevel', 'BEVEL'); bv.width = 0.004; bv.segments = 3
    return o


def mesh_obj(name, verts, faces, mat, smooth=True):
    me = bpy.data.meshes.new(name); me.from_pydata(T(verts).tolist(), [], faces.tolist()); me.update()
    if smooth:
        me.shade_smooth()
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); o.data.materials.append(mat)
    return o


def set_verts(o, V):
    o.data.vertices.foreach_set('co', T(V).astype(np.float32).ravel()); o.data.update()


def points_obj(name, n, radius, mat):
    me = bpy.data.meshes.new(name); me.vertices.add(n); me.attributes.new('err', 'FLOAT', 'POINT')
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o)
    gn = o.modifiers.new('pts', 'NODES'); tree = bpy.data.node_groups.new('pts', 'GeometryNodeTree'); gn.node_group = tree
    tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = tree.nodes.new('NodeGroupInput'); go = tree.nodes.new('NodeGroupOutput'); N = tree.nodes.new; L = tree.links.new
    mp = N('GeometryNodeMeshToPoints'); mp.inputs['Radius'].default_value = radius
    sm = N('GeometryNodeSetMaterial'); sm.inputs['Material'].default_value = mat
    L(gi.outputs[0], mp.inputs['Mesh'])
    if SURFACE:  # same surface construction for every method: points -> volume (sphere radius ~ own spacing) -> mesh
        pv = N('GeometryNodePointsToVolume'); pv.resolution_mode = 'VOXEL_SIZE'
        pv.inputs['Voxel Size'].default_value = 0.0015; pv.inputs['Radius'].default_value = radius; pv.inputs['Density'].default_value = 1.0
        vm = N('GeometryNodeVolumeToMesh'); vm.inputs['Threshold'].default_value = 0.5
        L(mp.outputs['Points'], pv.inputs['Points']); L(pv.outputs['Volume'], vm.inputs['Volume'])
        sn = N('GeometryNodeSampleNearest'); L(mp.outputs['Points'], sn.inputs['Geometry'])
        si = N('GeometryNodeSampleIndex'); si.data_type = 'FLOAT'; si.domain = 'POINT'
        na = N('GeometryNodeInputNamedAttribute'); na.data_type = 'FLOAT'; na.inputs['Name'].default_value = 'err'
        L(mp.outputs['Points'], si.inputs['Geometry']); L(na.outputs['Attribute'], si.inputs['Value']); L(sn.outputs['Index'], si.inputs['Index'])
        st = N('GeometryNodeStoreNamedAttribute'); st.data_type = 'FLOAT'; st.domain = 'POINT'; st.inputs['Name'].default_value = 'err'
        pos = N('GeometryNodeInputPosition'); bl = N('GeometryNodeBlurAttribute'); bl.data_type = 'FLOAT_VECTOR'
        bl.inputs['Iterations'].default_value = 12; L(pos.outputs['Position'], bl.inputs['Value'])
        sp_ = N('GeometryNodeSetPosition'); L(vm.outputs['Mesh'], sp_.inputs['Geometry']); L(bl.outputs['Value'], sp_.inputs['Position'])
        L(sp_.outputs['Geometry'], st.inputs['Geometry']); L(si.outputs['Value'], st.inputs['Value'])
        ss = N('GeometryNodeSetShadeSmooth'); L(st.outputs['Geometry'], ss.inputs['Geometry']); L(ss.outputs['Geometry'], sm.inputs['Geometry'])
        o['pv'] = pv.name; o['mp'] = mp.name
    else:
        L(mp.outputs['Points'], sm.inputs['Geometry'])
    L(sm.outputs['Geometry'], go.inputs[0])
    return o


def set_radius(o, r):
    t = o.modifiers['pts'].node_group
    t.nodes[o['mp']].inputs['Radius'].default_value = r
    if 'pv' in o:
        t.nodes[o['pv']].inputs['Radius'].default_value = r


def resize_points(o, n):
    me = bpy.data.meshes.new('pts'); me.vertices.add(n); me.attributes.new('err', 'FLOAT', 'POINT'); o.data = me


def set_points(o, P, e):
    o.data.vertices.foreach_set('co', T(P).astype(np.float32).ravel()); o.data.attributes['err'].data.foreach_set('value', e.astype(np.float32)); o.data.update()


def sphere_obj(name, r, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=64, ring_count=32); o = bpy.context.object; o.name = name
    bpy.ops.object.shade_smooth(); o.data.materials.append(mat); return o


def T(a):
    a = np.asarray(a, float); return np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1)


X = Z['X']; U = Z['U']; tris = Z['tris']
lo, hi = X.min(0), X.max(0); ctr = (lo+hi)/2; size = float((hi-lo).max())
M_REF = mat_principled('ref', '#9fb4c7', rough=.55, sss=.15)
M_GHOST = mat_principled('ghost', '#8f949c', rough=.6, alpha=.12)
M_ERR = mat_error(float(Z['vmax']))
M_RIG = mat_principled('rig', '#4a4d52', rough=.45, metal=.2)
M_FLOOR = mat_principled('floor', '#e6e4df', rough=.85)
M_STEEL = mat_principled('steel', '#c9ccd1', rough=.18, metal=1.)
M_STEEL_G = mat_principled('steelg', '#9aa0a8', rough=.3, metal=.5, alpha=.25)

# fixtures
if SCEN == 'S2':
    floor_y = lo[1]
    add_box('table', [lo[0]-4, floor_y-.03, lo[2]-4], [hi[0]+4, floor_y, hi[2]+4], M_FLOOR)
else:  # cantilever: clamp block around x <= -0.20, floor 0.25 m below
    add_box('clamp', [lo[0]-.03, lo[1]-.04, lo[2]-.04], [-0.20, hi[1]+.04, hi[2]+.04], M_RIG)
    add_box('post', [lo[0]-.03, 0.30, lo[2]-.04], [-0.20, lo[1]-.04, hi[2]+.04], M_RIG)
    floor_y = 0.30
    add_box('floor', [lo[0]-4, floor_y-.02, lo[2]-4], [hi[0]+4, floor_y, hi[2]+4], M_FLOOR)

surf = mesh_obj('surface', X, tris, M_REF)
pts = points_obj('points', len(Z['m0_pos'][0]), 0.0022 if SCEN != 'S2' else 0.0018, M_ERR)
strk_true = sphere_obj('striker_true', 0.03, M_STEEL) if 'sphere' in Z.files else None
strk_meth = sphere_obj('striker_meth', 0.03, M_STEEL) if 'sphere' in Z.files else None

# load glyph: arrow (tip at the loaded region, along the known force direction, length ~ ramp) and press pad
ARROW = None; PAD = None
if 'arrow_tip' in Z.files:
    M_ARW = mat_principled('arrow', '#2b2d31', rough=.35)
    L_SH, L_HEAD, R_SH, R_HEAD = (0.15, 0.045, 0.007, 0.017) if SCEN == 'S1' else (0.09, 0.03, 0.0045, 0.011)
    bpy.ops.mesh.primitive_cylinder_add(radius=R_SH, depth=L_SH, location=(0, 0, -L_HEAD-L_SH/2), vertices=32); sh = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=R_HEAD, radius2=0, depth=L_HEAD, location=(0, 0, -L_HEAD/2), vertices=32); hd = bpy.context.object
    bpy.ops.object.select_all(action='DESELECT'); sh.select_set(True); hd.select_set(True); bpy.context.view_layer.objects.active = sh
    bpy.ops.object.join(); ARROW = bpy.context.object; ARROW.data.materials.clear(); ARROW.data.materials.append(M_ARW)
    bpy.ops.object.shade_smooth()
    dv = T(Z['arrow_dir']); dv = dv/np.linalg.norm(dv)
    ARROW.rotation_euler = Vector(dv).to_track_quat('Z', 'Y').to_euler()     # local +Z = force direction; tip at origin
    if 'pad_half' in Z.files:
        h = float(Z['pad_half'])
        bpy.ops.mesh.primitive_cube_add(); PAD = bpy.context.object; PAD.scale = (h, h, 0.002); PAD.data.materials.append(M_RIG)
        bv = PAD.modifiers.new('bevel', 'BEVEL'); bv.width = 0.0015; bv.segments = 2


def place_glyphs(k):
    if ARROW is None:
        return
    s_ = float(Z['arrow_scale'][k]); tip = T(Z['arrow_tip'][k])
    ARROW.hide_render = s_ < 0.02
    ARROW.location = tuple(tip); ARROW.scale = (1, 1, max(s_, 0.02))
    if PAD is not None:
        PAD.location = tuple(tip-np.array([0, 0, 0.002]))


# lights
def light(name, kind, loc, energy, size_, rot_to):
    L = bpy.data.lights.new(name, kind); L.energy = energy
    if kind == 'AREA':
        L.size = size_
    loc, rot_to = T(loc), T(rot_to)
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (Vector(rot_to)-Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
light('key', 'AREA', ctr+np.array([.4, 1.0, .8]), 45, .9, ctr)
light('fill', 'AREA', ctr+np.array([-.9, .5, .6]), 12, 1.4, ctr)
light('rim', 'AREA', ctr+np.array([.2, .6, -1.0]), 25, .9, ctr)

# camera: 3/4 view from +z (the striker side), slightly above
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50
if SCEN == 'S2':
    eye = ctr+np.array([.22, .26, .42])*size/0.24
    look = ctr+np.array([0, -.01, 0])
else:
    eye = ctr+np.array([.42, .16, .92])
    look = ctr+np.array([.03, -.05, 0])
eye, look = T(eye), T(look)
cam.location = eye; cam.rotation_euler = (Vector(look)-Vector(eye)).to_track_quat('-Z', 'Y').to_euler()
cam.data.dof.use_dof = False

OUTDIR.mkdir(parents=True, exist_ok=True)
for pnl in PANELS:
    ref = pnl == 0
    surf.data.materials[0] = M_REF if ref else M_GHOST
    pts.hide_render = ref
    if not ref:
        resize_points(pts, Z[f'm{pnl-1}_pos'].shape[1])
        set_radius(pts, 0.8*float(Z[f'm{pnl-1}_spacing']) if SURFACE else 0.0022)
    if strk_true is not None:
        strk_true.data.materials[0] = M_STEEL if ref else M_STEEL_G
        strk_meth.hide_render = ref or f'm{pnl-1}_sphere' not in Z.files
    for k in FRAMES:
        set_verts(surf, X+U[k])
        if not ref:
            set_points(pts, Z[f'm{pnl-1}_pos'][k], Z[f'm{pnl-1}_err'][k])
        if strk_true is not None:
            strk_true.location = tuple(T(Z['sphere'][k]))
            if not strk_meth.hide_render:
                strk_meth.location = tuple(T(Z[f'm{pnl-1}_sphere'][k]))
        place_glyphs(k)
        sc.render.filepath = str(OUTDIR/f'p{pnl}'/f'{k:03d}.png')
        bpy.ops.render.render(write_still=True)
        print(f'panel {pnl} frame {k}', flush=True)
