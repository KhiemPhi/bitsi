"""Matplotlib rendering of BITSI results + a small auto-refreshing web page (no ROS dependency).

WebView(dir, port).add(request_id, frame_id, objects) renders one PNG per request (per object: 3D view, top view x-y,
side view x-z, coloured by segment, centroids marked) and rewrites index.html with the newest request first.
The HTTP server serves only that directory; it binds to all interfaces, so expose the port only on a trusted network.
"""
import html, json, os, threading, time
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CMAP = plt.get_cmap('tab10')


def image_panel(ax, o, margin=.6):
    """Camera image of the capture cropped around the detector's 2D box (green), with the segmentation mask tinted."""
    img = o.get('image'); box = o.get('bbox2d'); mask = o.get('mask2d'); ax.axis('off')
    if img is None: ax.set_title('no camera image'); return
    H, W = img.shape[:2]; ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    if mask is not None and mask.shape[:2] == (H, W):
        tint = np.zeros((H, W, 4)); tint[mask] = (0, 1, 0, .35); ax.imshow(tint)
    if box is not None:
        x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec='lime', lw=2))
        ax.set_xlim(max(0, x0 - margin * w), min(W, x1 + margin * w)); ax.set_ylim(min(H, y1 + margin * h), max(0, y0 - margin * h))
        ax.set_title(f"2D detection: box {int(w)}x{int(h)} px")
    else: ax.set_title('2D detection: no box in message')


def render_salsa(path, o):
    """SALSA plan: BITSI v4 parts (grasp part red), screw axis (black), best candidate's pose path with approach (z) arrows."""
    sp = o['salsa']; parts = sp['parts']; G = sp.get('grasp_part_points'); cands = sp.get('candidates') or []
    allp = np.concatenate([p['points'] for p in parts]); lo, hi = allp.min(0), allp.max(0)
    poses = cands[0]['poses'] if cands else []
    if poses: P = np.array([q['position'] for q in poses]); lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.1 + 1e-3
    fig = plt.figure(figsize=(15, 4.8)); ax3 = fig.add_subplot(1, 3, 1, projection='3d'); axs = [fig.add_subplot(1, 3, k) for k in (2, 3)]
    def R_of(q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    for j, p in enumerate(parts):
        q = p['points'][::max(1, len(p['points']) // 3000)]; col = CMAP(j % 10)
        ax3.scatter(q[:, 0], q[:, 1], q[:, 2], s=1, color=col, alpha=.5, label=f"{p['name']} {p['kind']} ({p['n_points']})")
        for ax, (a, b) in zip(axs, ((0, 1), (0, 2))): ax.scatter(q[:, a], q[:, b], s=1, color=col, alpha=.4)
    if G is not None and len(G):
        ax3.scatter(G[:, 0], G[:, 1], G[:, 2], s=4, color='red', label='grasp part')
        for ax, (a, b) in zip(axs, ((0, 1), (0, 2))): ax.scatter(G[:, a], G[:, b], s=4, color='red')
    sc = sp.get('screw')
    if sc:
        ax_, pt = np.array(sc['axis']), np.array(sc['point']); L = np.array([pt - ax_ * h, pt + ax_ * h])
        ax3.plot(L[:, 0], L[:, 1], L[:, 2], 'k-', lw=2, label='screw axis')
        for ax, (a, b) in zip(axs, ((0, 1), (0, 2))): ax.plot(L[:, a], L[:, b], 'k-', lw=1.5)
    if poses:
        ax3.plot(P[:, 0], P[:, 1], P[:, 2], '-', color='tab:purple', lw=1.5, label='EE path (candidate 0)')
        for q, pp in zip(poses, P):
            zz = R_of(q['quaternion_xyzw'])[:, 2] * h * .25; m = 'o' if q['gripper'] in ('open', 'closed') else 's'
            ax3.quiver(pp[0], pp[1], pp[2], zz[0], zz[1], zz[2], color='tab:purple', lw=.8)
            for ax, (a, b) in zip(axs, ((0, 1), (0, 2))):
                ax.plot([pp[a]], [pp[b]], m, color='tab:purple', ms=4); ax.arrow(pp[a], pp[b], zz[a] * .6, zz[b] * .6, color='tab:purple', width=h * .003)
        for q, pp in zip(poses, P):
            if q['name'] in ('pre_grasp', 'grasp', 'pre_press', 'touch', 'release'):
                for ax, (a, b) in zip(axs, ((0, 1), (0, 2))): ax.annotate(q['name'], (pp[a], pp[b]), fontsize=7)
    ax3.set_xlim(c[0] - h, c[0] + h); ax3.set_ylim(c[1] - h, c[1] + h); ax3.set_zlim(c[2] - h, c[2] + h); ax3.legend(fontsize=6, markerscale=4, loc='upper left')
    ax3.set_title(f"SALSA '{sp.get('command')}': {sc['task'] if sc else 'no screw'}" + (f", {len(cands)} candidate(s)" if cands else ''))
    for ax, (a, b), t in zip(axs, ((0, 1), (0, 2)), ('top view (x-y)', 'side view (x-z)')):
        ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel('xyz'[a] + ' [m]'); ax.set_ylabel('xyz'[b] + ' [m]'); ax.set_title(t); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def render(path, request_id, frame_id, objects):
    """objects: [{'label', 'id', 'segments': [{'name', 'centroid', ...}], 'seg_points': [(n,3)], 'raw_points': (N,3) input cloud}]"""
    objs = [o for o in objects if o.get('seg_points')]
    if not objs: return False
    ncol = 5 if any(o.get('image') is not None for o in objs) else 4
    fig = plt.figure(figsize=(5 * ncol, 4.8 * len(objs)))
    for r, o in enumerate(objs):
        P = o['seg_points']; raw = o.get('raw_points'); raw = None if raw is None or len(raw) == 0 else np.asarray(raw)
        allp = np.concatenate(P + ([raw] if raw is not None else [])); lo, hi = allp.min(0), allp.max(0); c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.1 + 1e-3
        k0 = ncol - 4
        if k0: image_panel(fig.add_subplot(len(objs), ncol, ncol * r + 1), o)
        axr = fig.add_subplot(len(objs), ncol, ncol * r + k0 + 1, projection='3d'); ax3 = fig.add_subplot(len(objs), ncol, ncol * r + k0 + 2, projection='3d')
        axs = [fig.add_subplot(len(objs), ncol, ncol * r + k0 + k) for k in (3, 4)]
        if raw is not None:   # full input cloud as received from the pipeline, before voxel/outlier filtering
            q = raw[::max(1, len(raw) // 6000)]
            axr.scatter(q[:, 0], q[:, 1], q[:, 2], s=1.5, c=q[:, 2], cmap='viridis')
            for ax, (a, b) in zip(axs, ((0, 1), (0, 2))): ax.scatter(q[:, a], q[:, b], s=4, color='0.85', zorder=0)
        axr.set_title(f"input cloud: {0 if raw is None else len(raw)} points"); axr.set_xlabel('x'); axr.set_ylabel('y'); axr.set_zlabel('z')
        for ax in (axr, ax3): ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
        for j, (p, s) in enumerate(zip(P, o['segments'])):
            col = CMAP(j % 10); step = max(1, len(p) // 3000); q = p[::step]
            ax3.scatter(q[:, 0], q[:, 1], q[:, 2], s=1.5, color=col, label=f"{s['name']} ({s['n_points']})")
            for ax, (a, b) in zip(axs, ((0, 1), (0, 2))):
                ax.scatter(q[:, a], q[:, b], s=1.5, color=col); m = s['centroid']
                ax.scatter([m[a]], [m[b]], s=60, color=col, edgecolor='k', marker='X', zorder=3); ax.annotate(s['name'].replace('segment_', 's'), (m[a], m[b]), fontsize=8)
        ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z'); ax3.legend(fontsize=7, markerscale=5, loc='upper left')
        ax3.set_title(f"{o.get('label') or 'object'} (id {o['id']}): BITSI {o.get('version', 'v0')}, {len(P)} segments")
        for ax, (a, b), t in zip(axs, ((0, 1), (0, 2)), ('top view (x-y), input in grey', 'side view (x-z), input in grey')):
            ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel('xyz'[a] + ' [m]'); ax.set_ylabel('xyz'[b] + ' [m]'); ax.set_title(t); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)   # request id and frame are in the page heading
    for o in objs:
        if o.get('salsa') and o['salsa'].get('parts'):
            render_salsa(path.replace('.png', f"_salsa_{o['id']}.png"), o)
    return True


def salsa_summary(o, png, d):
    sp = o.get('salsa')
    if not sp: return None
    img = png.replace('.png', f"_salsa_{o['id']}.png") if png else None
    if img and not os.path.exists(os.path.join(d, img)): img = None
    c0 = (sp.get('candidates') or [None])[0]
    return dict(command=sp.get('command'), error=sp.get('error'), screw=sp.get('screw'), png=img, n_cand=len(sp.get('candidates') or []),
                poses=c0['poses'] if c0 else [])


class WebView:
    def __init__(self, directory, port, keep=20, host='0.0.0.0', on_target=None, target='', on_version=None, version='v0',
                 on_salsa=None, salsa=''):
        self.dir = directory; self.keep = keep; self.entries = []; self.on_target = on_target; self.target = target
        self.on_version = on_version; self.version = version; self.on_salsa = on_salsa; self.salsa = salsa
        os.makedirs(directory, exist_ok=True); self.write_index()
        self.server = None
        if port:
            self.server = ThreadingHTTPServer((host, port), partial(QuietHandler, directory=directory, view=self))
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def set_target(self, target):
        self.target = target; self.write_index()

    def set_version(self, version):
        self.version = version; self.write_index()

    def set_salsa(self, salsa):
        self.salsa = salsa; self.write_index()

    def add(self, request_id, frame_id, objects):
        safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(request_id))
        png = f'req_{safe}.png'
        if not render(os.path.join(self.dir, png), request_id, frame_id, objects): png = None
        rows = [dict(id=o['id'], label=o.get('label', ''), error=o.get('error'), seconds=o.get('seconds'), n_points=o.get('n_points'),
                     segments=o.get('segments', []), version=o.get('version', 'v0'), salsa=salsa_summary(o, png, self.dir)) for o in objects]
        self.entries.insert(0, dict(request_id=str(request_id), frame_id=frame_id, time=time.strftime('%H:%M:%S'), png=png, objects=rows))
        for old in self.entries[self.keep:]:
            for f in [old['png']] + [o['salsa']['png'] for o in old['objects'] if o.get('salsa') and o['salsa'].get('png')]:
                if not f: continue
                try: os.remove(os.path.join(self.dir, f))
                except OSError: pass
        self.entries = self.entries[:self.keep]; self.write_index()

    def write_index(self):
        e = html.escape; parts = ['<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">',
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>BITSI results</title><style>',
            'body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}img{max-width:100%;border:1px solid #ddd;background:#fff}',
            'table{border-collapse:collapse;margin:6px 0 18px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#eee}',
            '.err{color:#b00}section{margin-bottom:28px}</style></head><body><h1>BITSI Chebyshev segments</h1>',
            '<p>Newest request first; refreshes every 5 s. Coordinates in the cloud frame (robot frame).</p>',
            '<form action="target" method="get">Run BITSI on: <input name="q" size="30" placeholder="all objects" value="' + e(self.target) + '"> ',
            '<button>Set</button> <small>label text, comma-separated for several (e.g. <i>cup, red bottle</i>); empty = all objects</small></form>',
            f"<p>Current target: <b>{e(self.target) or 'all objects'}</b></p>",
            '<form action="version" method="get">BITSI version: <select name="v">'
            + ''.join(f'<option value="{v}"{" selected" if v == self.version else ""}>{v}</option>' for v in ('v0', 'v3', 'v4', 'v5', 'v6'))
            + '</select> <button>Use</button> <small>v0 = main.py (default); v3-v6 = fitpart BITSI (v5: partial-cloud candidate; v6 experimental). '
              'Changing it re-runs the last request.</small></form>',
            '<form action="salsa" method="get">SALSA command: <input name="c" size="30" placeholder="off (segmentation only)" value="'
            + e(self.salsa) + '"> <button>Plan</button> <small>task [amount] [reverse]: e.g. <i>lift 10 cm</i>, <i>turn 90 deg</i>, '
              '<i>hinge</i>, <i>slide</i>, <i>press</i>, <i>pivot</i>, <i>handover</i>; empty = off. Re-plans the last request.</small></form>']
        if not self.entries: parts.append('<p>No results yet: trigger a detection in the perception pipeline.</p>')
        for en in self.entries:
            parts.append(f"<section><h2>request {e(en['request_id'])} <small>{e(en['time'])}, frame {e(str(en['frame_id']))}</small></h2>")
            if en['png']: parts.append(f"<img src=\"{e(en['png'])}?t={e(en['time'])}\" alt=\"BITSI segments\">")
            for o in en['objects']:
                parts.append(f"<h3>{e(str(o['label']) or 'object')} (id {e(str(o['id']))}): {o.get('n_points')} points, {o.get('seconds')} s, BITSI {e(str(o.get('version', 'v0')))}</h3>")
                if o['error']: parts.append(f"<p class='err'>{e(o['error'])}</p>"); continue
                parts.append('<table><tr><th>segment</th><th>points</th><th>centroid x, y, z [m]</th><th>OBB extent [m]</th><th>IBR</th></tr>')
                for s in o['segments']:
                    f3 = lambda v: ', '.join(f'{x:.3f}' for x in v) if v else ''
                    ibr = '' if s.get('ibr_ratio') is None else '%.3f' % s['ibr_ratio']
                    parts.append(f"<tr><td>{e(s['name'])}</td><td>{s['n_points']}</td><td>{f3(s.get('centroid'))}</td><td>{f3(s.get('obb_extent'))}</td><td>{ibr}</td></tr>")
                parts.append('</table>')
                sa = o.get('salsa')
                if sa:
                    parts.append(f"<h4>SALSA: {e(str(sa['command']))}</h4>")
                    if sa['error']: parts.append(f"<p class='err'>{e(sa['error'])}</p>")
                    if sa['png']: parts.append(f"<img src=\"{e(sa['png'])}?t={e(en['time'])}\" alt=\"SALSA plan\">")
                    sc = sa['screw']
                    if sc:
                        amt = f"{sc['theta']:.3f} m" if sc['translation'] else f"{np.degrees(sc['theta']):.1f} deg"
                        parts.append(f"<p>screw <b>{e(sc['task'])}</b>: axis {', '.join('%.3f' % v for v in sc['axis'])}, point "
                                     f"{', '.join('%.3f' % v for v in sc['point'])}, {'translation' if sc['translation'] else 'rotation'} {amt}"
                                     f"{', lift %.2f m first' % sc['lift'] if sc['lift'] else ''}; grasp part {sc['grasp_points']} points; "
                                     f"{sa['n_cand']} candidate pose sequence(s) on /salsa/plan, best on /salsa/poses</p>")
                    if sa['poses']:
                        parts.append('<table><tr><th>#</th><th>pose</th><th>position x, y, z [m]</th><th>quaternion x, y, z, w</th><th>gripper</th></tr>')
                        for k, q in enumerate(sa['poses']):
                            parts.append(f"<tr><td>{k}</td><td>{e(q['name'])}</td><td>{', '.join('%.3f' % v for v in q['position'])}</td>"
                                         f"<td>{', '.join('%.3f' % v for v in q['quaternion_xyzw'])}</td><td>{e(q['gripper'])}</td></tr>")
                        parts.append('</table>')
            parts.append('</section>')
        parts.append('</body></html>')
        tmp = os.path.join(self.dir, '.index.tmp')
        with open(tmp, 'w') as f: f.write('\n'.join(parts))
        os.replace(tmp, os.path.join(self.dir, 'index.html'))


class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, view=None, **k):
        self.view = view; super().__init__(*a, **k)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path == '/salsa' and self.view is not None:
            c = parse_qs(u.query).get('c', [''])[0]
            if self.view.on_salsa: self.view.on_salsa(c)
            else: self.view.set_salsa(c)
            self.send_response(303); self.send_header('Location', '/'); self.end_headers(); return
        if u.path == '/version' and self.view is not None:
            v = parse_qs(u.query).get('v', ['v0'])[0]
            if self.view.on_version: self.view.on_version(v)
            else: self.view.set_version(v)
            self.send_response(303); self.send_header('Location', '/'); self.end_headers(); return
        if u.path == '/target' and self.view is not None:
            q = parse_qs(u.query).get('q', [''])[0]
            if self.view.on_target: self.view.on_target(q)
            else: self.view.set_target(q)
            self.send_response(303); self.send_header('Location', '/'); self.end_headers(); return
        return super().do_GET()

    def log_message(self, *a): pass
