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


def render(path, request_id, frame_id, objects):
    """objects: [{'label', 'id', 'segments': [{'name', 'centroid', ...}], 'seg_points': [ (n,3) arrays ]}]"""
    objs = [o for o in objects if o.get('seg_points')]
    if not objs: return False
    fig = plt.figure(figsize=(15, 4.6 * len(objs)))
    for r, o in enumerate(objs):
        P = o['seg_points']; allp = np.concatenate(P); lo, hi = allp.min(0), allp.max(0); c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.1 + 1e-3
        ax3 = fig.add_subplot(len(objs), 3, 3 * r + 1, projection='3d'); axs = [fig.add_subplot(len(objs), 3, 3 * r + k) for k in (2, 3)]
        for j, (p, s) in enumerate(zip(P, o['segments'])):
            col = CMAP(j % 10); step = max(1, len(p) // 3000); q = p[::step]
            ax3.scatter(q[:, 0], q[:, 1], q[:, 2], s=1.5, color=col, label=f"{s['name']} ({s['n_points']})")
            for ax, (a, b) in zip(axs, ((0, 1), (0, 2))):
                ax.scatter(q[:, a], q[:, b], s=1.5, color=col); m = s['centroid']
                ax.scatter([m[a]], [m[b]], s=60, color=col, edgecolor='k', marker='X', zorder=3); ax.annotate(s['name'].replace('segment_', 's'), (m[a], m[b]), fontsize=8)
        ax3.set_xlim(c[0] - h, c[0] + h); ax3.set_ylim(c[1] - h, c[1] + h); ax3.set_zlim(c[2] - h, c[2] + h)
        ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z'); ax3.legend(fontsize=7, markerscale=5, loc='upper left')
        ax3.set_title(f"{o.get('label') or 'object'} (id {o['id']}): {len(P)} segments")
        for ax, (a, b), t in zip(axs, ((0, 1), (0, 2)), ('top view (x-y)', 'side view (x-z)')):
            ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel('xyz'[a] + ' [m]'); ax.set_ylabel('xyz'[b] + ' [m]'); ax.set_title(t); ax.grid(alpha=.3)
    fig.suptitle(f'BITSI request {request_id}  (frame {frame_id})'); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    return True


class WebView:
    def __init__(self, directory, port, keep=20, host='0.0.0.0'):
        self.dir = directory; self.keep = keep; self.entries = []; os.makedirs(directory, exist_ok=True); self.write_index()
        self.server = None
        if port:
            self.server = ThreadingHTTPServer((host, port), partial(QuietHandler, directory=directory))
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(self, request_id, frame_id, objects):
        safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(request_id))
        png = f'req_{safe}.png'
        if not render(os.path.join(self.dir, png), request_id, frame_id, objects): png = None
        rows = [dict(id=o['id'], label=o.get('label', ''), error=o.get('error'), seconds=o.get('seconds'), n_points=o.get('n_points'),
                     segments=o.get('segments', [])) for o in objects]
        self.entries.insert(0, dict(request_id=str(request_id), frame_id=frame_id, time=time.strftime('%H:%M:%S'), png=png, objects=rows))
        for old in self.entries[self.keep:]:
            if old['png']:
                try: os.remove(os.path.join(self.dir, old['png']))
                except OSError: pass
        self.entries = self.entries[:self.keep]; self.write_index()

    def write_index(self):
        e = html.escape; parts = ['<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">',
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>BITSI results</title><style>',
            'body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}img{max-width:100%;border:1px solid #ddd;background:#fff}',
            'table{border-collapse:collapse;margin:6px 0 18px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#eee}',
            '.err{color:#b00}section{margin-bottom:28px}</style></head><body><h1>BITSI Chebyshev segments</h1>',
            '<p>Newest request first; refreshes every 5 s. Coordinates in the cloud frame (robot frame).</p>']
        if not self.entries: parts.append('<p>No results yet: trigger a detection in the perception pipeline.</p>')
        for en in self.entries:
            parts.append(f"<section><h2>request {e(en['request_id'])} <small>{e(en['time'])}, frame {e(str(en['frame_id']))}</small></h2>")
            if en['png']: parts.append(f"<img src=\"{e(en['png'])}?t={e(en['time'])}\" alt=\"BITSI segments\">")
            for o in en['objects']:
                parts.append(f"<h3>{e(str(o['label']) or 'object')} (id {e(str(o['id']))}): {o.get('n_points')} points, {o.get('seconds')} s</h3>")
                if o['error']: parts.append(f"<p class='err'>{e(o['error'])}</p>"); continue
                parts.append('<table><tr><th>segment</th><th>points</th><th>centroid x, y, z [m]</th><th>OBB extent [m]</th><th>IBR</th></tr>')
                for s in o['segments']:
                    f3 = lambda v: ', '.join(f'{x:.3f}' for x in v) if v else ''
                    parts.append(f"<tr><td>{e(s['name'])}</td><td>{s['n_points']}</td><td>{f3(s.get('centroid'))}</td><td>{f3(s.get('obb_extent'))}</td><td>{s.get('ibr_ratio', 0):.3f}</td></tr>")
                parts.append('</table>')
            parts.append('</section>')
        parts.append('</body></html>')
        tmp = os.path.join(self.dir, '.index.tmp')
        with open(tmp, 'w') as f: f.write('\n'.join(parts))
        os.replace(tmp, os.path.join(self.dir, 'index.html'))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
