import os
import re
import numpy as np
import pandas as pd
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo
import math
import json

class IMUViewer(Gtk.Window):
    def __init__(self):
        super().__init__(title="IMU Data Viewer")
        self.set_default_size(1200, 800)
        
        # Load data
        self.load_data()
        
        # View parameters
        self.zoom = 1.0
        self.pan_x = 0.0
        self.dragging = False
        self.last_mouse_x = 0
        self.highlight_center = None
        self.window_size = 100
        self.half_window = self.window_size // 2
        
        # Selection points system
        self.selection_points = []  # List of {'index': int, 'label': str, 'timestamp': int}
        self.selection_file = os.path.join('snippets', 'selection_points.json')
        
        # Keyboard shortcuts mapping - use letters instead of numbers
        self.label_shortcuts = {
            Gdk.KEY_s: "STANDING",
            Gdk.KEY_w: "WALKING", 
            Gdk.KEY_r: "RUNNING",
            Gdk.KEY_p: "PASS",
            Gdk.KEY_h: "SHOOT"  # Changed from Gdk.KEY_t to Gdk.KEY_h
        }
        
        # Colors
        self.colors = {
            'mag_a': (0.0, 0.4, 0.8),
            'mag_g': (1.0, 0.5, 0.0),
            'highlight': (1.0, 0.0, 0.0),
            'grid': (0.8, 0.8, 0.8),
            'bg': (1.0, 1.0, 1.0),
            'selection_points': {
                'STANDING': (0.5, 0.0, 0.5),
                'WALKING': (0.0, 0.8, 0.0),
                'RUNNING': (0.8, 0.8, 0.0),
                'PASS': (0.0, 0.6, 0.8),
                'SHOOT': (0.8, 0.2, 0.2)
            }
        }
        
        # Performance optimization
        self.last_draw_time = 0
        self.min_draw_interval = 33  # ~30fps limit for better performance
        self.redraw_scheduled = False
        
        self.setup_ui()
        self.load_selection_points()
        
    def load_data(self):
        # Load IMU data
        imu_file = 'football_data/imu_data_20250529_183017.txt'
        # imu_file = 'football_data/imu_data_20250529_181854.txt' # LEFT
        
        imu_rows = []
        with open(imu_file, 'r') as f:
            for line in f:
                parts = re.split(r'\s*-\s*', line.strip(), maxsplit=1)
                if len(parts) == 2:
                    try:
                        ts = int(parts[0].strip())
                        vals = list(map(float, parts[1].strip().split()))
                        if len(vals) == 6:
                            imu_rows.append([ts] + vals)
                    except ValueError:
                        continue
        
        self.imu_df = pd.DataFrame(
            imu_rows,
            columns=['Timestamp','accx','accy','accz','gyrox','gyroy','gyroz']
        )
        self.imu_df['mag_A'] = np.sqrt(
            self.imu_df['accx']**2 + self.imu_df['accy']**2 + self.imu_df['accz']**2
        )
        self.imu_df['mag_G'] = np.sqrt(
            self.imu_df['gyrox']**2 + self.imu_df['gyroy']**2 + self.imu_df['gyroz']**2
        )
        
        # Normalize timestamps to 0-1 range for easier drawing
        self.timestamps = self.imu_df['Timestamp'].values
        self.t_min, self.t_max = self.timestamps.min(), self.timestamps.max()
        self.t_norm = (self.timestamps - self.t_min) / (self.t_max - self.t_min)
        
        # Store raw data for saving
        self.raw_six = self.imu_df[['accx','accy','accz','gyrox','gyroy','gyroz']].values
        
        # Create save directory
        self.save_root = 'snippets'
        os.makedirs(self.save_root, exist_ok=True)
        
        # Load events data
        self.load_events()
        
    def load_events(self):
        """Load and process events data"""
        events_file = 'football_data/events_183017.txt'
        # events_file = 'football_data/events_181854.txt' # LEFT
        event_data = []
        event_types_of_interest = [
            'INIZIO_CAMMINATA','FINE_CAMMINATA','PASSAGGIO','TIRO',
            'INIZIO_CORSA','FINE_CORSA','INIZIO_STANDING','FINE_STANDING'
        ]
        
        with open(events_file, 'r') as f:
            for line in f:
                match = re.match(r'(\d+) - (.*)', line.strip())
                if match:
                    try:
                        ts = int(match.group(1))
                        action = match.group(2).strip()
                        event_data.append([ts, action])
                    except ValueError:
                        continue
        
        self.event_df = pd.DataFrame(event_data, columns=['Timestamp','Action'])
        self.event_df_filtered = self.event_df[self.event_df['Action'].isin(event_types_of_interest)].copy()
        
        # Normalize event timestamps to same 0-1 range as IMU data
        self.event_df_filtered['t_norm'] = (self.event_df_filtered['Timestamp'] - self.t_min) / (self.t_max - self.t_min)
        
    def load_selection_points(self):
        """Load previously saved selection points"""
        if os.path.exists(self.selection_file):
            try:
                with open(self.selection_file, 'r') as f:
                    self.selection_points = json.load(f)
                print(f"Loaded {len(self.selection_points)} selection points from {self.selection_file}")
            except Exception as e:
                print(f"Error loading selection points: {e}")
                self.selection_points = []
        else:
            self.selection_points = []
            print(f"No existing selection points file found at {self.selection_file}")
            
    def save_selection_points(self):
        """Save selection points to file"""
        try:
            os.makedirs(os.path.dirname(self.selection_file), exist_ok=True)
            with open(self.selection_file, 'w') as f:
                json.dump(self.selection_points, f, indent=2)
            print(f"Saved {len(self.selection_points)} selection points to {self.selection_file}")
        except Exception as e:
            print(f"Error saving selection points: {e}")
        
    def setup_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)
        
        # Drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect('draw', self.on_draw)
        
        # Enable hardware acceleration for better performance
        self.drawing_area.set_app_paintable(True)
            
        self.drawing_area.set_size_request(1200, 600)
        self.drawing_area.connect('motion-notify-event', self.on_motion)
        self.drawing_area.connect('button-press-event', self.on_click)
        self.drawing_area.connect('button-release-event', self.on_button_release)
        self.drawing_area.connect('scroll-event', self.on_scroll)
        
        # Enable keyboard events at window level
        self.set_can_focus(True)
        self.connect('key-press-event', self.on_key_press)
        
        # Enable events
        self.drawing_area.set_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.SCROLL_MASK
        )
        
        vbox.pack_start(self.drawing_area, True, True, 0)
        
        # Status bar
        self.status_label = Gtk.Label()
        status_text = "Keys S/W/R/P/H: Quick label (Standing/Walking/Running/Pass/Shoot) | Mouse: highlight, drag to pan, scroll to zoom | Right click: label dialog"
        self.status_label.set_text(status_text)
        vbox.pack_start(self.status_label, False, False, 5)
        
    def on_key_press(self, widget, event):
        """Handle keyboard shortcuts for labeling"""
        if event.keyval in self.label_shortcuts:
            if self.highlight_center is not None and \
               self.highlight_center >= self.half_window and \
               self.highlight_center <= len(self.imu_df) - self.half_window:
                
                label = self.label_shortcuts[event.keyval]
                self.save_snippet(label)
                self.add_selection_point(self.highlight_center, label)
                self.drawing_area.queue_draw()
                return True
        return False
        
    def add_selection_point(self, index, label):
        """Add a selection point and save to file"""
        timestamp = int(self.timestamps[index])
        
        # Remove existing point at same index if it exists
        self.selection_points = [p for p in self.selection_points if p['index'] != index]
        
        # Add new point
        new_point = {
            'index': index,
            'label': label,
            'timestamp': timestamp
        }
        self.selection_points.append(new_point)
        
        # Save to file
        self.save_selection_points()
        
        # Update status
        self.status_label.set_text(f"Marked point at index {index} with label '{label}' (saved to {self.selection_file})")
        GLib.timeout_add(3000, self.reset_status_text)
        
    def reset_status_text(self):
        """Reset status text to default"""
        self.status_label.set_text("Keys S/W/R/P/H: Quick label (Standing/Walking/Running/Pass/Shoot) | Mouse: highlight, drag to pan, scroll to zoom | Right click: label dialog")
        return False
        
    def on_draw(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        
        # Clear background
        cr.set_source_rgb(*self.colors['bg'])
        cr.paint()
        
        # Define plot areas (3 subplots)
        margin = 50
        plot_height = (height - 4 * margin) // 3
        plot_areas = [
            (margin, margin, width - 2*margin, plot_height),
            (margin, 2*margin + plot_height, width - 2*margin, plot_height),
            (margin, 3*margin + 2*plot_height, width - 2*margin, plot_height)
        ]
        
        # Draw plots
        self.draw_plot(cr, plot_areas[0], self.imu_df['mag_A'], "Magnitude A", self.colors['mag_a'])
        self.draw_plot(cr, plot_areas[1], self.imu_df['mag_G'], "Magnitude G", self.colors['mag_g'])
        self.draw_events_plot(cr, plot_areas[2])
        
        # Draw selection points markers
        self.draw_selection_points(cr, plot_areas[:2])
        
        # Draw highlight lines if active
        if self.highlight_center is not None:
            self.draw_highlight(cr, plot_areas[:2], width)
            
    def draw_selection_points(self, cr, plot_areas):
        """Draw markers for saved selection points"""
        if not self.selection_points:
            return
            
        visible_start_norm = -self.pan_x
        visible_end_norm = -self.pan_x + 1.0 / self.zoom
        
        for point in self.selection_points:
            index = point['index']
            label = point['label']
            
            if 0 <= index < len(self.t_norm):
                t_norm = self.t_norm[index]
                
                # Only draw if visible
                if visible_start_norm <= t_norm <= visible_end_norm:
                    color = self.colors['selection_points'].get(label, (0.5, 0.5, 0.5))
                    
                    for area in plot_areas:
                        marker_x, _ = self.world_to_screen(t_norm, 0, area[2], area[3], area)
                        
                        if area[0] <= marker_x <= area[0] + area[2]:
                            # Draw vertical line
                            cr.set_source_rgba(*color, 0.8)
                            cr.set_line_width(3)
                            cr.move_to(marker_x, area[1])
                            cr.line_to(marker_x, area[1] + area[3])
                            cr.stroke()
                            
                            # Draw circle at top
                            cr.set_source_rgb(*color)
                            cr.arc(marker_x, area[1] + 10, 5, 0, 2 * math.pi)
                            cr.fill()
                            
                            # Draw label
                            cr.set_source_rgb(0, 0, 0)
                            cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                            cr.set_font_size(10)
                            
                            # Abbreviated label for space
                            short_label = label[:3]
                            text_extents = cr.text_extents(short_label)
                            
                            # White background for readability
                            cr.set_source_rgba(1, 1, 1, 0.9)
                            cr.rectangle(marker_x - text_extents.width/2 - 2, area[1] + 15, 
                                       text_extents.width + 4, text_extents.height + 4)
                            cr.fill()
                            
                            # Draw text
                            cr.set_source_rgb(0, 0, 0)
                            cr.move_to(marker_x - text_extents.width/2, area[1] + 15 + text_extents.height)
                            cr.show_text(short_label)
            
    def draw_plot(self, cr, area, data, title, color):
        x, y, w, h = area
        
        # Clip drawing to plot area for performance
        cr.save()
        cr.rectangle(x, y, w, h)
        cr.clip()
        
        # Enable Cairo optimizations
        cr.set_antialias(cairo.ANTIALIAS_FAST)
        
        # Draw border
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(1)
        cr.rectangle(x, y, w, h)
        cr.stroke()
        
        # Draw title
        cr.restore()  # Remove clip for title
        cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(14)
        text_extents = cr.text_extents(title)
        cr.move_to(x + w/2 - text_extents.width/2, y - 10)
        cr.show_text(title)
        
        # Re-clip for data drawing
        cr.save()
        cr.rectangle(x, y, w, h)
        cr.clip()
        
        # Normalize data
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        if data_range == 0:
            data_range = 1
        data_norm = (data - data_min) / data_range
        
        # Calculate visible range
        visible_start_norm = -self.pan_x
        visible_end_norm = -self.pan_x + 1.0 / self.zoom
        
        # Find indices for visible data
        start_idx = np.searchsorted(self.t_norm, visible_start_norm)
        end_idx = np.searchsorted(self.t_norm, visible_end_norm)
        start_idx = max(0, start_idx - 1)
        end_idx = min(len(self.t_norm), end_idx + 1)
        
        # Add decimation when zoomed out for performance
        visible_points = end_idx - start_idx
        max_points = w * 2  # 2 points per pixel max
        
        if visible_points > max_points:
            # Smart decimation: keep local min/max to preserve signal shape
            indices = self.smart_decimate(start_idx, end_idx, max_points, data_norm)
        else:
            indices = range(start_idx, end_idx)
        
        # Draw data line
        cr.set_source_rgb(*color)
        cr.set_line_width(1)
        
        # Use Cairo path for better performance
        cr.new_path()
        first_point = True
        for i in indices:
            t, d = self.t_norm[i], data_norm.iloc[i] if hasattr(data_norm, 'iloc') else data_norm[i]
            screen_x, screen_y = self.world_to_screen(t, d, w, h, area)
            
            if first_point:
                cr.move_to(screen_x, screen_y)
                first_point = False
            else:
                cr.line_to(screen_x, screen_y)
        
        cr.stroke()
        cr.restore()
        
    def smart_decimate(self, start_idx, end_idx, max_points, data_norm):
        """Smart decimation that preserves local min/max to maintain signal shape"""
        total_points = end_idx - start_idx
        if total_points <= max_points:
            return range(start_idx, end_idx)
        
        # Calculate window size for local min/max detection (more aggressive)
        window_size = max(2, total_points // (max_points // 2))
        
        selected_indices = []
        
        for i in range(start_idx, end_idx, window_size):
            window_end = min(i + window_size, end_idx)
            window_data = data_norm.iloc[i:window_end] if hasattr(data_norm, 'iloc') else data_norm[i:window_end]
            
            if len(window_data) == 0:
                continue
                
            # Simpler: just take first, middle, and last point of each window
            selected_indices.append(i)
            if window_end - i > 2:
                selected_indices.append(i + (window_end - i) // 2)
            if window_end - 1 > i:
                selected_indices.append(window_end - 1)
        
        # Sort and remove duplicates
        return sorted(set(selected_indices))
        
    def draw_events_plot(self, cr, area):
        x, y, w, h = area
        
        # Clip to area
        cr.save()
        cr.rectangle(x, y, w, h)
        cr.clip()
        
        # Draw colored intervals for START/END pairs
        self.draw_event_intervals(cr, x, y, w, h)
        
        # Draw border
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(1)
        cr.rectangle(x, y, w, h)
        cr.stroke()
        
        cr.restore()
        
        # Draw title
        cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(14)
        text_extents = cr.text_extents("Events")
        cr.move_to(x + w/2 - text_extents.width/2, y - 10)
        cr.show_text("Events")
        
        # Clip again for event markers
        cr.save()
        cr.rectangle(x, y, w, h)
        cr.clip()
        
        # Draw event markers
        visible_start_norm = -self.pan_x
        visible_end_norm = -self.pan_x + 1.0 / self.zoom
        
        visible_events = self.event_df_filtered[
            (self.event_df_filtered['t_norm'] >= visible_start_norm) &
            (self.event_df_filtered['t_norm'] <= visible_end_norm)
        ]
        
        # Calculate zoom-responsive font size for event labels
        base_font_size = 8
        zoom_font_size = max(6, min(16, base_font_size + self.zoom * 2))
        
        for _, event in visible_events.iterrows():
            t_norm = event['t_norm']
            action = event['Action']
            
            # Transform to screen coordinates
            event_x = (t_norm + self.pan_x) * self.zoom * w + x
            
            if x <= event_x <= x + w:
                # Draw vertical line
                cr.set_source_rgb(0.8, 0.2, 0.2)
                cr.set_line_width(2)
                cr.move_to(event_x, y)
                cr.line_to(event_x, y + h)
                cr.stroke()
                
                # Draw label with zoom-responsive font
                cr.set_source_rgb(0, 0, 0)
                cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(zoom_font_size)
                
                # Use high-quality text rendering
                cr.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
                
                cr.save()
                cr.translate(event_x + 2, y + h - 5)
                cr.rotate(-math.pi/2)
                
                # Add text outline for better readability
                text_path = cr.text_path(action)
                cr.set_source_rgb(1, 1, 1)  # White outline
                cr.set_line_width(2)
                cr.stroke_preserve()
                cr.set_source_rgb(0, 0, 0)  # Black text
                cr.fill()
                
                cr.restore()
        
        cr.restore()
        
    def draw_event_intervals(self, cr, x, y, w, h):
        """Draw colored intervals between INIZIO/FINE event pairs"""
        # Define colors for different event types
        event_colors = {
            'CAMMINATA': (0.2, 0.8, 0.2, 0.3),  # Green
            'CORSA': (0.8, 0.6, 0.2, 0.3),      # Orange
            'STANDING': (0.2, 0.2, 0.8, 0.3),   # Blue
        }
        
        # Calculate zoom-responsive font size for interval labels
        interval_font_size = max(8, min(20, 10 + self.zoom * 3))
        
        # Group events by type
        for event_type, color in event_colors.items():
            start_events = self.event_df_filtered[
                self.event_df_filtered['Action'] == f'INIZIO_{event_type}'
            ]['t_norm'].values
            
            end_events = self.event_df_filtered[
                self.event_df_filtered['Action'] == f'FINE_{event_type}'
            ]['t_norm'].values
            
            # Pair up start and end events
            for start_t in start_events:
                # Find the next end event after this start
                end_candidates = end_events[end_events > start_t]
                if len(end_candidates) > 0:
                    end_t = end_candidates[0]
                    
                    # Transform to screen coordinates
                    start_x = (start_t + self.pan_x) * self.zoom * w + x
                    end_x = (end_t + self.pan_x) * self.zoom * w + x
                    
                    # Only draw if interval is visible
                    if end_x >= x and start_x <= x + w:
                        # Clamp to visible area
                        draw_start_x = max(start_x, x)
                        draw_end_x = min(end_x, x + w)
                        
                        # Draw colored rectangle
                        cr.set_source_rgba(*color)
                        cr.rectangle(draw_start_x, y, draw_end_x - draw_start_x, h)
                        cr.fill()
                        
                        # Draw label in the middle if interval is wide enough
                        interval_width = draw_end_x - draw_start_x
                        min_width_for_label = max(30, 100 / self.zoom)  # Adaptive minimum width
                        if interval_width > min_width_for_label:
                            cr.set_source_rgb(0, 0, 0)
                            cr.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                            cr.set_font_size(interval_font_size)
                            
                            # Use high-quality text rendering
                            cr.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
                            
                            text_extents = cr.text_extents(event_type)
                            label_x = draw_start_x + (interval_width - text_extents.width) / 2
                            label_y = y + h / 2 + text_extents.height / 2
                            
                            # Add text background for better readability
                            padding = 2
                            cr.set_source_rgba(1, 1, 1, 0.8)  # Semi-transparent white background
                            cr.rectangle(label_x - padding, label_y - text_extents.height - padding,
                                       text_extents.width + 2*padding, text_extents.height + 2*padding)
                            cr.fill()
                            
                            # Draw text
                            cr.set_source_rgb(0, 0, 0)
                            cr.move_to(label_x, label_y)
                            cr.show_text(event_type)
        
    def draw_highlight(self, cr, plot_areas, width):
        if self.highlight_center < self.half_window or \
           self.highlight_center > len(self.imu_df) - self.half_window:
            return
            
        left_idx = self.highlight_center - self.half_window
        right_idx = self.highlight_center + self.half_window - 1
        
        left_t = self.t_norm[left_idx]
        right_t = self.t_norm[right_idx]
        
        cr.set_source_rgba(*self.colors['highlight'], 0.7)
        cr.set_line_width(2)
        cr.set_dash([5, 5])
        
        for area in plot_areas:
            left_x, _ = self.world_to_screen(left_t, 0, area[2], area[3], area)
            right_x, _ = self.world_to_screen(right_t, 0, area[2], area[3], area)
            
            # Draw vertical lines
            cr.move_to(left_x, area[1])
            cr.line_to(left_x, area[1] + area[3])
            cr.move_to(right_x, area[1])
            cr.line_to(right_x, area[1] + area[3])
            
        cr.stroke()
        cr.set_dash([])
        
    def world_to_screen(self, x, y, width, height, plot_area):
        """Convert normalized coordinates to screen coordinates"""
        x_screen = (x + self.pan_x) * self.zoom * width + plot_area[0]
        y_screen = plot_area[1] + (1.0 - y) * plot_area[3]
        return x_screen, y_screen
        
    def screen_to_world(self, x_screen, y_screen, width, height, plot_area):
        """Convert screen coordinates to normalized coordinates"""
        x_norm = (x_screen - plot_area[0]) / (width * self.zoom) - self.pan_x
        y_norm = 1.0 - (y_screen - plot_area[1]) / plot_area[3]
        return x_norm, y_norm
        
    def find_nearest_index(self, x_norm):
        """Find closest data point index"""
        if x_norm < 0 or x_norm > 1:
            return None
        diffs = np.abs(self.t_norm - x_norm)
        return int(diffs.argmin())
        
    def on_motion(self, widget, event):
        import time
        current_time = time.time() * 1000
        
        # Aggressive frame rate limiting
        if self.redraw_scheduled or current_time - self.last_draw_time < self.min_draw_interval:
            return
        
        if self.dragging:
            # Pan the view
            dx = (event.x - self.last_mouse_x) / (widget.get_allocated_width() - 100)
            self.pan_x += dx / self.zoom
            # Allow more liberal panning
            self.pan_x = max(-10.0, min(10.0, self.pan_x))
            self.last_mouse_x = event.x
            self.schedule_redraw(widget)
            return
            
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        
        # Check if mouse is in plottable area
        margin = 50
        plot_height = (height - 4 * margin) // 3
        
        old_highlight = self.highlight_center
        
        if event.y < margin or event.y > 3*margin + 2*plot_height:
            self.highlight_center = None
        else:
            area = (margin, margin, width - 2*margin, plot_height)
            x_norm, _ = self.screen_to_world(event.x, event.y, area[2], area[3], area)
            idx = self.find_nearest_index(x_norm)
            
            if idx is not None:
                self.highlight_center = idx
            else:
                self.highlight_center = None
                
        # Only redraw if highlight changed
        if old_highlight != self.highlight_center:
            self.schedule_redraw(widget)
        
    def schedule_redraw(self, widget):
        """Schedule a redraw with throttling"""
        if not self.redraw_scheduled:
            self.redraw_scheduled = True
            GLib.idle_add(self._do_redraw, widget)
            
    def _do_redraw(self, widget):
        """Actual redraw function called from idle"""
        import time
        self.last_draw_time = time.time() * 1000
        self.redraw_scheduled = False
        widget.queue_draw()
        return False  # Don't repeat
        
    def on_click(self, widget, event):
        if event.button == 1:  # Left click
            self.dragging = True
            self.last_mouse_x = event.x
            return
            
        if self.highlight_center is None:
            return
            
        if self.highlight_center < self.half_window or \
           self.highlight_center > len(self.imu_df) - self.half_window:
            return
            
        # Show label selection dialog
        self.show_label_dialog()
        
    def on_button_release(self, widget, event):
        if event.button == 1:  # Left click release
            if not self.dragging:
                # This was a click, not a drag - handle snippet selection
                if self.highlight_center is not None and \
                   self.highlight_center >= self.half_window and \
                   self.highlight_center <= len(self.imu_df) - self.half_window:
                    self.show_label_dialog()
            self.dragging = False
        
    def show_label_dialog(self):
        labels = ["STANDING", "WALKING", "RUNNING", "PASS", "SHOOT"]
        dialog = Gtk.Dialog(title="Choose Label", parent=self, flags=0)
        
        for idx, label in enumerate(labels):
            dialog.add_button(label, idx)
            
        dialog.set_default_size(400, 100)
        response = dialog.run()
        
        if 0 <= response < len(labels):
            label = labels[response]
            self.save_snippet(label)
            self.add_selection_point(self.highlight_center, label)
            self.drawing_area.queue_draw()
            
        dialog.destroy()
        
    def save_snippet(self, label):
        folder = os.path.join(self.save_root, label)
        os.makedirs(folder, exist_ok=True)
        
        snippet = self.raw_six[
            self.highlight_center - self.half_window:
            self.highlight_center + self.half_window
        ]
        
        existing = [f for f in os.listdir(folder) if f.endswith('.npy')]
        fname = os.path.join(folder, f"{len(existing):04d}.npy")
        np.save(fname, snippet)
        
        print(f"Saved snippet under label '{label}' → {fname}")
        
    def on_scroll(self, widget, event):
        # Convert mouse position to world coordinates before zoom
        width = widget.get_allocated_width()
        margin = 50
        plot_width = width - 2 * margin
        
        # Mouse position in plot coordinates (0-1)
        mouse_plot_x = (event.x - margin) / plot_width
        
        # Current world coordinate at mouse position
        world_x_at_mouse = mouse_plot_x / self.zoom - self.pan_x
        
        old_zoom = self.zoom
         
        # Zoom functionality
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom *= 1.1
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom /= 1.1
            
        # Reasonable zoom limits
        max_zoom = 100.0
        self.zoom = max(0.1, min(max_zoom, self.zoom))
        
        # Adjust pan so the world coordinate under the mouse stays the same
        new_pan_x = mouse_plot_x / self.zoom - world_x_at_mouse
        self.pan_x = new_pan_x
        
        # Clamp pan
        self.pan_x = max(-10.0, min(10.0, self.pan_x))
        
        self.schedule_redraw(widget)

def main():
    app = IMUViewer()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    
    # Focus the window for keyboard events
    app.grab_focus()
    
    Gtk.main()

if __name__ == "__main__":
    main()
