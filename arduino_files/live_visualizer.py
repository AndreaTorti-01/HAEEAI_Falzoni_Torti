import threading, serial, time, json
from datetime import datetime
import os # For os.getcwd and os.path.join

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk
import cairo

# --- CONFIG ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SLIDING_WINDOW = 50

# --- DATA & LABELS ---
# Revert to Standing at the bottom
class_labels = ['Standing','Walking','Running','Pass!','Shot!']
label_map    = {'S': 'Standing', 'W': 'Walking', 'R': 'Running', 'P': 'Pass!', 'H': 'Shot!'}
class_to_idx = {lbl: i for i, lbl in enumerate(class_labels)}

class LiveDataRecorder:
    def __init__(self):
        self.data = []  # List of (timestamp, class_label)
        self.running = False
        self.ser = None
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            # Skip first dirty line
            self.ser.readline()
        except Exception as e:
            print('Serial open error:', e)
            self.running = False
            return
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.thread = None

    def _read_loop(self):
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    raw_line = self.ser.readline()
                    if raw_line: # Check if any data was read
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                        now = datetime.now()
                        unix_time_ms = int(now.timestamp() * 1000)
                        if line: # Check if line is not empty after strip
                            code = line[0]
                            class_label = label_map.get(code)
                            if class_label:
                                self.data.append((unix_time_ms, class_label))
                    # If raw_line is empty, it could be a timeout, just loop again
                else:
                    # self.ser is None or closed, and self.running is True, wait a bit
                    if self.running: # Check again before sleep, as running might have changed
                        time.sleep(0.1)
            except serial.SerialException as se:
                # Handle cases where port is closed or other serial errors specifically
                if self.running: # Only print if we are supposed to be running
                    print(f"SerialException in read loop: {se}")
                time.sleep(0.5) # Longer sleep on serial exception
            except Exception as e:
                if self.running:
                    # More robust error printing
                    print(f"Unexpected error in read loop: Type={type(e).__name__}, Msg='{str(e)}'")
                time.sleep(0.1)

    def get_recent(self, n):
        return self.data[-n:]

    def export_json(self, path):
        # Convert tuples to lists for JSON compatibility
        with open(path, 'w') as f:
            json.dump([list(item) for item in self.data], f)

    def export_subtitles(self, path):
        if not self.data:
            return
        start0 = self.data[0][0]
        with open(path, 'w') as f:
            for i, (ts, label) in enumerate(self.data):
                rel_start = (ts - start0) / 1000.0
                rel_end   = (ts + 1000 - start0) / 1000.0
                def fmt(t):
                    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t - int(t))*1000)
                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                f.write(f"{i+1}\n{fmt(rel_start)} --> {fmt(rel_end)}\n{label}\n\n")

# --- GUI ---
class LiveVisualizer(Gtk.Window):
    def __init__(self, recorder):
        super().__init__(title="Live Arduino Data Visualizer")
        self.recorder = recorder
        self.set_default_size(800, 400)
        self.connect("destroy", lambda w: (self.recorder.stop(), Gtk.main_quit()))

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        # Drawing area for Cairo
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(800, 300)
        self.drawing_area.connect("draw", self.on_draw)
        vbox.pack_start(self.drawing_area, True, True, 0)

        # Controls
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.start_btn = Gtk.Button(label="Start")
        self.start_btn.connect("clicked", lambda b: (self.recorder.start(), self.start_btn.set_sensitive(False), self.stop_btn.set_sensitive(True)))
        hbox.pack_start(self.start_btn, False, False, 0)

        self.stop_btn = Gtk.Button(label="Stop")
        self.stop_btn.set_sensitive(False)
        self.stop_btn.connect("clicked", lambda b: (self.recorder.stop(), self.start_btn.set_sensitive(True), self.stop_btn.set_sensitive(False)))
        hbox.pack_start(self.stop_btn, False, False, 0)

        btn_export_json = Gtk.Button(label="Export as JSON")
        btn_export_json.connect("clicked", self.on_export_json)
        hbox.pack_start(btn_export_json, False, False, 0)

        btn_export_srt = Gtk.Button(label="Export as Subtitles")
        btn_export_srt.connect("clicked", self.on_export_srt)
        hbox.pack_start(btn_export_srt, False, False, 0)

        btn_quit = Gtk.Button(label="Quit")
        btn_quit.connect("clicked", lambda b: (self.recorder.stop(), Gtk.main_quit()))
        hbox.pack_end(btn_quit, False, False, 0)

        vbox.pack_start(hbox, False, False, 0)
        self.add(vbox)
        self.show_all()

        GLib.timeout_add(200, self.update_plot)

    def update_plot(self):
        self.drawing_area.queue_draw()
        return True

    def _get_theme_colors(self, widget):
        style_context = widget.get_style_context()
        # Use get_property to avoid deprecation warnings
        bg_color_gdk = style_context.get_property("background-color", Gtk.StateFlags.NORMAL)
        fg_color_gdk = style_context.get_property("color", Gtk.StateFlags.NORMAL)

        # Convert Gdk.RGBA to Cairo-compatible (r, g, b, a) tuples
        bg_cairo = (bg_color_gdk.red, bg_color_gdk.green, bg_color_gdk.blue, bg_color_gdk.alpha)
        fg_cairo = (fg_color_gdk.red, fg_color_gdk.green, fg_color_gdk.blue, fg_color_gdk.alpha)
        
        # Determine if background is dark for grid lines
        # Simple luminance check: (0.299*R + 0.587*G + 0.114*B)
        luminance = 0.299 * bg_cairo[0] + 0.587 * bg_cairo[1] + 0.114 * bg_cairo[2]
        is_dark_theme = luminance < 0.5
        
        grid_color_cairo = (fg_cairo[0], fg_cairo[1], fg_cairo[2], 0.2) # Semi-transparent foreground for grid

        return bg_cairo, fg_cairo, grid_color_cairo, is_dark_theme

    def on_draw(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        margin = 80
        bar_width = max((width - 2*margin) / SLIDING_WINDOW, 1)
        bar_gap = 2

        bg_color, fg_color, grid_color, is_dark_theme = self._get_theme_colors(widget)

        # Use theme background color
        cr.set_source_rgba(*bg_color)
        cr.paint()

        data = self.recorder.get_recent(SLIDING_WINDOW)
        total = len(self.recorder.data)
        pad = SLIDING_WINDOW - len(data)
        if total < SLIDING_WINDOW:
            x_indices = list(range(SLIDING_WINDOW))
        else:
            x_indices = list(range(total-SLIDING_WINDOW, total))
        if pad > 0:
            labels = [None]*pad + [lbl for _,lbl in data]
        else:
            labels = [lbl for _,lbl in data]

        # Colors for each class (these are data-specific, not theme-specific)
        palette = [
            (0.6,0.6,0.6), # fallback gray
            (0.2,0.5,0.9), # Standing
            (0.2,0.8,0.2), # Walking
            (0.9,0.7,0.2), # Running
            (0.9,0.2,0.2), # Pass!
            (0.6,0.2,0.8), # Shot!
        ]

        # Calculate positions for class rows
        plot_height = height - 2*margin
        class_height = plot_height / len(class_labels)
        
        # Draw horizontal grid lines and labels (Standing at bottom)
        cr.set_source_rgba(*grid_color) # Use theme-aware grid color
        cr.set_line_width(1)
        for idx, label_text in enumerate(class_labels):
            # Y position from bottom up: Standing (idx 0) is lowest.
            y_center_of_row = (height - margin) - (idx * class_height) - (class_height / 2)
            
            # Draw grid line
            cr.move_to(margin, y_center_of_row)
            cr.line_to(width - margin, y_center_of_row)
            cr.stroke()
            
            # Draw label
            cr.set_source_rgba(*fg_color) # Use theme foreground color
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(12)
            text_extents = cr.text_extents(label_text)
            cr.move_to(margin - text_extents.width - 10, y_center_of_row + text_extents.height/2)
            cr.show_text(label_text)

        # Draw bars centered on class positions
        for i, lbl in enumerate(labels):
            x_bar_start = margin + i * bar_width
            if lbl is not None:
                class_idx = class_to_idx[lbl]
                # The color palette index is class_idx + 1 because palette[0] is fallback gray
                color_idx_in_palette = class_idx + 1 
                if color_idx_in_palette >= len(palette): color_idx_in_palette = 0 # Fallback if out of bounds
                cr.set_source_rgb(*palette[color_idx_in_palette])
                
                # Center bar on class position (Standing at bottom)
                y_center_of_row = (height - margin) - (class_idx * class_height) - (class_height / 2)
                bar_visual_height = class_height * 0.6  # Make bars 60% of class height
                y_bar_top = y_center_of_row - bar_visual_height / 2
                
                cr.rectangle(x_bar_start, y_bar_top, bar_width - bar_gap, bar_visual_height)
                cr.fill()
            else:
                # Draw small gray bar at bottom for no data
                cr.set_source_rgb(*palette[0]) # palette[0] is fallback gray
                cr.rectangle(x_bar_start, height - margin - 10, bar_width - bar_gap, 10)
                cr.fill()

        # Draw axes
        cr.set_source_rgba(*fg_color) # Use theme foreground color
        cr.set_line_width(2)
        cr.move_to(margin, margin)
        cr.line_to(margin, height - margin)
        cr.line_to(width - margin, height - margin)
        cr.stroke()

        # Draw title
        cr.set_font_size(16)
        text_extents = cr.text_extents("Live Class Recognition")
        cr.move_to(width/2 - text_extents.width/2, margin - 20)
        cr.show_text("Live Class Recognition")

    def on_export_json(self, _):
        filename = "live_data.json"
        file_path = os.path.join(os.getcwd(), filename)
        try:
            self.recorder.export_json(file_path)
            print(f"JSON data saved to {file_path}")
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")
            # Optionally, show a Gtk.MessageDialog for critical errors if console is not monitored
            # For now, just printing to console as per "no prompts"

    def on_export_srt(self, _):
        filename = "live_subtitles.srt"
        file_path = os.path.join(os.getcwd(), filename)
        try:
            self.recorder.export_subtitles(file_path)
            print(f"Subtitles saved to {file_path}")
        except Exception as e:
            print(f"Error saving subtitles to {file_path}: {e}")

if __name__ == '__main__':
    recorder = LiveDataRecorder()
    LiveVisualizer(recorder)
    Gtk.main()
