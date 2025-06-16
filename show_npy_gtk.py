import os, json
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

class SnippetBrowser(Gtk.Window):
    def __init__(self):
        super().__init__(title="Snippet Browser")
        self.set_default_size(800, 600)
        self.snippets_root = "snippets"
        self.selection_file = os.path.join(self.snippets_root, "selection_points.json")
        self.load_selection()
        self.build_ui()
        self.rebuild_index()
        self.show_snippet(0)

    def load_selection(self):
        if os.path.exists(self.selection_file):
            with open(self.selection_file,'r') as f:
                self.points = json.load(f)
        else:
            self.points = []

    def save_selection(self):
        with open(self.selection_file,'w') as f:
            json.dump(self.points, f, indent=2)

    def rebuild_index(self):
        # group by label, sort by timestamp to match file ordering
        grouped = {}
        for p in self.points:
            grouped.setdefault(p['label'], []).append(p)
        self.entries = []
        for label, pts in grouped.items():
            pts.sort(key=lambda x:x['timestamp'])
            for idx, p in enumerate(pts):
                fname = f"{idx:04d}.npy"
                path = os.path.join(self.snippets_root, label, fname)
                self.entries.append({'label':label,'path':path,'timestamp':p['timestamp']})
        self.entries.sort(key=lambda e:e['timestamp'])
        # filtered view
        self.current_filter = "All"
        self.filtered = list(self.entries)
        self.cur = 0
        # compute a single global max over magnitudes for all snippets
        self.compute_global_max()

    def compute_global_max(self):
        max_val = 0
        for ent in self.entries:
            try:
                data = np.load(ent['path'])
                mag_A = np.linalg.norm(data[:, :3], axis=1)
                mag_G = np.linalg.norm(data[:, 3:], axis=1)
                val = max(mag_A.max(), mag_G.max())
                if val > max_val:
                    max_val = val
            except Exception:
                continue
        self.global_max = max_val

    def build_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
        # controls
        ctrl = Gtk.Box(spacing=6)
        vbox.pack_start(ctrl, False, False, 0)
        # label filter
        self.filter_cb = Gtk.ComboBoxText()
        self.filter_cb.append_text("All")
        labels = sorted({p['label'] for p in self.points})
        for L in labels: self.filter_cb.append_text(L)
        self.filter_cb.set_active(0)
        self.filter_cb.connect("changed", self.on_filter)
        ctrl.pack_start(self.filter_cb, False, False, 0)
        # Prev / Next
        btn_prev = Gtk.Button(label="Previous")
        btn_prev.connect("clicked", lambda w: self.navigate(-1))
        ctrl.pack_start(btn_prev, False, False, 0)
        btn_next = Gtk.Button(label="Next")
        btn_next.connect("clicked", lambda w: self.navigate(1))
        ctrl.pack_start(btn_next, False, False, 0)
        # Delete
        btn_del = Gtk.Button(label="Delete")
        btn_del.connect("clicked", lambda w: self.delete_current())
        ctrl.pack_start(btn_del, False, False, 0)
        # add a label for keyboard shortcuts
        self.shortcut_label = Gtk.Label(
            label="Shortcuts: ←/→ prev/next | Canc delete | Esc exit"
        )
        vbox.pack_start(self.shortcut_label, False, False, 0)

        # add status label for snippet position
        self.status_label = Gtk.Label(label="")
        vbox.pack_start(self.status_label, False, False, 0)

        # figure
        self.fig = Figure(figsize=(5,3))
        self.canvas = FigureCanvas(self.fig)
        vbox.pack_start(self.canvas, True, True, 0)
        # key events
        self.connect("key-press-event", self.on_key)

    def on_filter(self, cb):
        self.current_filter = cb.get_active_text()
        if self.current_filter=="All":
            self.filtered = self.entries[:]
        else:
            self.filtered = [e for e in self.entries if e['label']==self.current_filter]
        self.cur = 0
        self.show_snippet(self.cur)

    def navigate(self, delta):
        if not self.filtered: return
        self.cur = (self.cur + delta) % len(self.filtered)
        self.show_snippet(self.cur)

    def delete_current(self):
        if not self.filtered: return
        ent = self.filtered[self.cur]
        # remove file
        try: os.remove(ent['path'])
        except: pass
        # remove JSON entry by timestamp+label
        self.points = [p for p in self.points
                       if not (p['label']==ent['label'] and p['timestamp']==ent['timestamp'])]
        self.save_selection()
        # renumber remaining .npy in that label folder
        folder = os.path.dirname(ent['path'])
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        for i,f in enumerate(files):
            if f != f"{i:04d}.npy":
                os.rename(os.path.join(folder,f), os.path.join(folder,f"{i:04d}.npy"))
        # rebuild index and view
        self.rebuild_index()
        self.on_filter(self.filter_cb)

    def show_snippet(self, idx):
        self.fig.clf()
        if not self.filtered:
            return
        ent = self.filtered[idx]
        data = np.load(ent['path'])

        # compute magnitudes
        mag_A = np.linalg.norm(data[:, :3], axis=1)
        mag_G = np.linalg.norm(data[:, 3:], axis=1)

        ax = self.fig.add_subplot(111)
        ax.plot(mag_A, label="Mag A")
        ax.plot(mag_G, label="Mag G")
        # apply fixed scale
        if getattr(self, 'global_max', 0) > 0:
            ax.set_ylim(-self.global_max, self.global_max)

        ax.set_title(f"{ent['label']} @ {ent['timestamp']}")
        ax.legend()

        # update status label: “snippet current/total”
        total = len(self.filtered)
        self.status_label.set_text(f"Snippet {idx+1}/{total}")

        self.canvas.draw()

    def on_key(self, widget, event):
        key = event.keyval
        handled = False

        if key in (Gdk.KEY_Left,):
            self.navigate(-1)
            handled = True
        elif key in (Gdk.KEY_Right,):
            self.navigate(1)
            handled = True
        elif key == Gdk.KEY_Delete:               # Canc key
            self.delete_current()
            handled = True
        elif key in (Gdk.KEY_d, Gdk.KEY_D):       # letter 'd'
            self.delete_current()
            handled = True
        elif key == Gdk.KEY_Escape:                # exit
            Gtk.main_quit()
            handled = True
        elif key == Gdk.KEY_Up:
            # cycle filter up
            n = self.filter_cb.get_model().iter_n_children(None)
            idx = (self.filter_cb.get_active() - 1) % n
            self.filter_cb.set_active(idx)
            handled = True
        elif key == Gdk.KEY_Down:
            # cycle filter down
            n = self.filter_cb.get_model().iter_n_children(None)
            idx = (self.filter_cb.get_active() + 1) % n
            self.filter_cb.set_active(idx)
            handled = True

        if handled:
            return True   # stop propagation to avoid moving focus
        return False

def main():
    win = SnippetBrowser()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()

if __name__=="__main__":
    main()
