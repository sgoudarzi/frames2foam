"""Export an animated mesh as a per-frame OBJ sequence with consistent vertex order,
ready for `frames2foam check` / `frames2foam write`.

Runs inside Blender (tested with 4.0), which reads .blend, .fbx, .abc (Alembic),
.glb/.gltf and .usd. Shape keys, armatures and modifiers are evaluated, so what is
written is exactly what plays in the viewport.

    blender -b --python blender_export_frames.py -- \
        --input heart.fbx --objects LV RV --out frames/ --start 1 --end 30

    blender -b heart.blend --python blender_export_frames.py -- --objects Heart --out frames/

Options
    --input FILE        model to import (.fbx .abc .glb .gltf .usd .usdc .usda); omit to use the open .blend
    --objects NAME ...  mesh objects to export (default: every mesh with animation or deformation)
    --out DIR           output directory
    --start/--end N     frame range (default: the scene range, or for an imported file the
                        range of its animation); --step N to subsample
    --join              write all objects into one surface per frame instead of one sequence each
    --scale S           multiply coordinates (e.g. 0.01 for centimetres to metres)
    --local             write object-space coordinates instead of world space

Writes DIR/<name>_<frame>.obj and DIR/manifest.json (frames, fps, suggested period,
vertex counts). The run stops with an error if the vertex count of an object changes
between frames, since frames2foam needs vertex correspondence; disable modifiers that
change topology (remesh, decimate, adaptive subdivision) before exporting.
"""
import json
import os
import sys

import bpy


def parse_args():
    import argparse
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='blender_export_frames.py')
    ap.add_argument('--input')
    ap.add_argument('--objects', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int)
    ap.add_argument('--end', type=int)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--join', action='store_true')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--local', action='store_true')
    return ap.parse_args(argv)


def import_model(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == '.abc':
        bpy.ops.wm.alembic_import(filepath=path, as_background_job=False)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext in ('.usd', '.usda', '.usdc', '.usdz'):
        bpy.ops.wm.usd_import(filepath=path)
    else:
        raise SystemExit(f'unsupported input format {ext}')
    return bpy.context.scene


def animation_range(objs):
    """Frame range covered by the actions driving these objects (FBX import leaves the
    scene at its default range, so the animation range is taken from the actions)."""
    lo, hi = None, None
    for o in objs:
        for owner in (o, o.data.shape_keys, o.parent):
            ad = getattr(owner, 'animation_data', None) if owner is not None else None
            if ad and ad.action:
                a, b = ad.action.frame_range
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        for m in o.modifiers:
            if m.type == 'ARMATURE' and m.object and m.object.animation_data and m.object.animation_data.action:
                a, b = m.object.animation_data.action.frame_range
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
    return (None, None) if lo is None else (int(round(lo)), int(round(hi)))


def is_animated(obj):
    if obj.animation_data and obj.animation_data.action:
        return True
    if obj.data.shape_keys and obj.data.shape_keys.animation_data:
        return True
    if any(m.type in ('ARMATURE', 'MESH_SEQUENCE_CACHE', 'MESH_CACHE') for m in obj.modifiers):
        return True
    return obj.parent is not None and obj.parent.animation_data is not None


def evaluated_triangles(obj, depsgraph, world, scale):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        M = ev.matrix_world if world else None
        verts = [tuple(scale * c for c in ((M @ v.co) if M is not None else v.co)) for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
    finally:
        ev.to_mesh_clear()
    return verts, tris


def write_obj(path, verts, tris, comment):
    with open(path, 'w') as f:
        f.write(f'# {comment}\n')
        f.writelines('v %.9g %.9g %.9g\n' % v for v in verts)
        f.writelines('f %d %d %d\n' % (a + 1, b + 1, c + 1) for a, b, c in tris)


def main():
    a = parse_args()
    scene = import_model(a.input) if a.input else bpy.context.scene
    meshes = [o for o in scene.objects if o.type == 'MESH']
    if a.objects:
        names = {o.name: o for o in meshes}
        missing = [n for n in a.objects if n not in names]
        if missing:
            raise SystemExit(f'objects not found: {missing}; mesh objects are {sorted(names)}')
        objs = [names[n] for n in a.objects]
    else:
        objs = [o for o in meshes if is_animated(o)] or meshes
    r0, r1 = animation_range(objs) if a.input else (None, None)
    start = a.start if a.start is not None else (r0 if r0 is not None else scene.frame_start)
    end = a.end if a.end is not None else (r1 if r1 is not None else scene.frame_end)
    frames = list(range(start, end + 1, a.step))
    fps = scene.render.fps / scene.render.fps_base
    os.makedirs(a.out, exist_ok=True)

    counts = {}
    width = len(str(frames[-1]))
    for f in frames:
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        joined_v, joined_t = [], []
        for o in objs:
            v, t = evaluated_triangles(o, dg, not a.local, a.scale)
            n0 = counts.setdefault(o.name, (len(v), len(t)))
            if n0 != (len(v), len(t)):
                raise SystemExit(f'{o.name}: topology changes at frame {f} ({n0} -> {(len(v), len(t))}); '
                                 'disable modifiers that remesh or decimate')
            if a.join:
                off = len(joined_v)
                joined_v += v
                joined_t += [(i + off, j + off, k + off) for i, j, k in t]
            else:
                write_obj(os.path.join(a.out, f'{o.name}_{f:0{width}d}.obj'), v, t,
                          f'{o.name} frame {f}, t = {(f - start) / fps:.6f} s')
        if a.join:
            write_obj(os.path.join(a.out, f'joined_{f:0{width}d}.obj'), joined_v, joined_t,
                      f'{"+".join(o.name for o in objs)} frame {f}, t = {(f - start) / fps:.6f} s')
        print(f'frame {f} written', flush=True)

    manifest = {
        'source': a.input or bpy.data.filepath,
        'objects': {n: {'vertices': c[0], 'triangles': c[1]} for n, c in counts.items()},
        'joined': a.join,
        'frames': frames,
        'fps': fps,
        'suggested_period_s': len(frames) * a.step / fps,
        'note': ('If the last frame repeats the first, frames2foam drops it automatically. '
                 'Otherwise the period is frames x step / fps.'),
        'scale': a.scale,
        'coordinates': 'object' if a.local else 'world',
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'exported {len(frames)} frames of {len(objs)} object(s) to {a.out}')


if __name__ == '__main__':
    main()
