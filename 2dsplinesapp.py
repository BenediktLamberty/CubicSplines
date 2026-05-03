from splineclasses import *
from splineclassesnd import *
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseEvent
import matplotlib.gridspec 
import matplotlib.figure 
import matplotlib.axes
import matplotlib.markers
import matplotlib.path
import matplotlib.transforms
from enum import Enum, auto


def safely_get_int(var: tk.IntVar, default: int = 0) -> int:
    try:
        return var.get()
    except tk.TclError:
        return default
    
def safely_get_float(var: tk.DoubleVar, default: float = 0) -> float:
    try:
        return var.get()
    except tk.TclError:
        return default
    
def restrict_to_interval(t: float, a: float, b: float) -> float:
    if t > b: return b 
    elif t < a: return a 
    else: return t
    
class Mode(Enum):
    NONE = auto()
    MOVE_T = auto()
    MOVE_XY = auto()
    CAR_T = auto()
    CAR_XY = auto()
    DRAW = auto()


class SplineType(Enum):
    CUBIC_SPLINE = "Kubische Spline"
    B_SPLINE = "B-Spline"
    CATMULL_ROM_SPLINE = "Catmull-Rom-Spline"


class SplineInterpolatingPointsApp:
    root: tk.Tk
    t_nodes: np.ndarray
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    t_graph: np.ndarray
    x_graph: np.ndarray
    y_graph: np.ndarray
    t_min: float = 0.0
    t_max: float = 1.0
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_car: float = 0.0
    car_pos: np.ndarray = np.zeros((4, 2))
    node_count: int
    spline_type: SplineType = SplineType.CUBIC_SPLINE
    spline_type_str: tk.StringVar
    periodic:      tk.BooleanVar
    speed:         tk.DoubleVar
    show_car:      tk.BooleanVar
    run_car:       tk.BooleanVar
    show_derivs:   tk.BooleanVar
    canvas:    FigureCanvasTkAgg
    figure:    matplotlib.figure.Figure
    gridspec:  matplotlib.gridspec.GridSpec
    main_axes: matplotlib.axes.Axes
    time_axes: matplotlib.axes.Axes
    draw_button: ttk.Button
    spline: CurveND | None = None
    dragging_index: int | None = None
    mode: Mode = Mode.NONE
    DENSITY: int = 1000
    EXCLUSION_DIST_XY: float = 0.02
    EXCLUSION_RATIO_T: float = 0.01
    FRAME_RATE: int = 60
    ARROW_SCALE: float = 0.3

    def __init__(self, root: tk.Tk):
        # Root
        self.root = root
        self.root.title("Zweidimensionale Kubische Splines")
        self.root.geometry('1000x600')

        #  Variablen
        self.spline_type_str = tk.StringVar(value=self.spline_type.value)
        self.periodic = tk.BooleanVar(value=False)
        self.speed = tk.DoubleVar(value=0.01)
        self.show_car = tk.BooleanVar(value=False)
        self.run_car = tk.BooleanVar(value=False)
        self.show_derivs = tk.BooleanVar(value=False)
        self.mode = Mode.NONE

        # Frames
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Control_Frame
        ttk.Label(control_frame, text="Zeichnen:").pack(anchor=tk.W)
        combobox = ttk.Combobox(control_frame, 
                     values=[value.value for value in SplineType], 
                     textvariable=self.spline_type_str, 
                     state="readonly")
        combobox.bind('<<ComboboxSelected>>', self.spline_type_changed)
        combobox.pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="periodische Spline",
                        variable=self.periodic,
                        command=self.periodic_changed).pack(anchor=tk.W)
        self.draw_button = ttk.Button(control_frame, 
                                      text="Neu Zeichnen",
                                      command=self.trigger_drawing_spline,
                                      width=15)
        self.draw_button.pack(anchor=tk.CENTER, pady=3)
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)


        ttk.Label(control_frame, text="Bewegbarer Punkt:").pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Punkt anzeigen",
                        variable=self.show_car,
                        command=self.spawn_car).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Vektoren anzeigen",
                        variable=self.show_derivs,
                        command=self.update_car).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Punkt bewegen",
                        variable=self.run_car,
                        command=self.update_car).pack(anchor=tk.W)
        slider_row = ttk.Frame(control_frame)
        slider_row.pack(anchor=tk.W)
        ttk.Label(slider_row, text="Geschwindigkeit: ").pack(side=tk.LEFT)
        ttk.Scale(slider_row, variable=self.speed, from_=0, to=0.02).pack(side=tk.LEFT)
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        
        
        # Canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.gridspec = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[20, 1])
        self.main_axes = self.figure.add_subplot(self.gridspec[0, 0])
        self.time_axes = self.figure.add_subplot(self.gridspec[1, 0])

        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.reset_points()
        self.root.after(1000 // self.FRAME_RATE, self.repeat)
        # update o.ä. TODO

    def spline_type_changed(self, *args):
        old_type = self.spline_type
        self.spline_type = SplineType(self.spline_type_str.get())
        if not self.periodic.get() and ((old_type == SplineType.B_SPLINE and self.spline_type != SplineType.B_SPLINE) or
                (old_type != SplineType.B_SPLINE and self.spline_type == SplineType.B_SPLINE)):
            # Konvertiere t von B -> normal oder normal -> B
            if self.node_count < self.get_min_node_count():
                self.reset_points()
            else:
                self.t_max = self.get_t_node_count() - 1
                self.t_nodes = np.linspace(self.t_min, self.t_max, self.get_t_node_count())
        self.update_spline()

    def periodic_changed(self, *args): # TODO für B-Spline
        # immer nicht-periodisch zeichnen
        if self.mode == Mode.DRAW or self.spline is None or self.node_count < self.get_min_node_count():
            self.periodic.set(False)
        # nicht B-Spline:
        elif self.spline_type != SplineType.B_SPLINE:
            # auf periodisch geschaltet
            if self.periodic.get(): 
                self.t_max = self.node_count
                self.t_nodes = np.append(self.t_nodes, self.t_max)
                self.update_spline()
            # auf nicht-periodisch geschaltet
            else:
                self.t_max = self.node_count - 1
                if self.t_nodes[-2] > self.t_max: # Zurücksetzen falls unschön 
                    self.t_nodes = np.linspace(self.t_min, self.t_max, self.node_count)
                else: # Sonst einfach letzten Zeitpunkt entfernen
                    self.t_nodes = self.t_nodes[:-1]
                    self.t_nodes[-1] = self.t_max
                self.update_spline()
        # B-Spline:
        elif self.spline_type == SplineType.B_SPLINE:
            # in jedem Fall resetten
            self.t_max = self.get_t_node_count() - 1
            self.t_nodes = np.linspace(self.t_min, self.t_max, self.get_t_node_count())
            self.update_spline()

    def get_t_node_count(self) -> int:
        if self.periodic.get(): return self.node_count + 1
        elif self.spline_type == SplineType.B_SPLINE: return max(self.node_count - 2, 0)
        else: return self.node_count

    def get_min_node_count(self) -> int:
        if self.spline_type == SplineType.B_SPLINE and not self.periodic.get(): return 4
        else: return 2

    def reset_points(self):
        self.t_nodes = np.array([])
        self.x_nodes = np.array([])
        self.y_nodes = np.array([])
        self.t_graph = np.array([])
        self.x_graph = np.array([])
        self.y_graph = np.array([])
        self.t_min = 0.0
        self.t_max = 1.0
        self.x_min = 0.0
        self.x_max = 2.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.node_count = 0
        self.spline = None
        self.periodic.set(False)
        self.update_spline()

    def update_spline(self):
        if self.node_count < self.get_min_node_count():
            self.t_graph = np.array([])
            self.x_graph = np.array([])
            self.y_graph = np.array([])
            self.spline = None
            self.update_car()
            return
        
        # Point-Set erstellen
        point_set: PointSetND | None = None
        if self.periodic.get():
            periodic_x_nodes = np.append(self.x_nodes, self.x_nodes[0])
            periodic_y_nodes = np.append(self.y_nodes, self.y_nodes[0])
            assert len(self.t_nodes) == len(periodic_x_nodes) and len(self.t_nodes) == len(periodic_y_nodes)
            point_set = PointSetND(2, self.t_nodes, np.array([periodic_x_nodes, periodic_y_nodes]))
        elif self.spline_type != SplineType.B_SPLINE:
            point_set = PointSetND(2, self.t_nodes, np.array([self.x_nodes, self.y_nodes]))

        # Natürliche Kubische Spline
        if self.spline_type == SplineType.CUBIC_SPLINE and not self.periodic.get():
            assert point_set is not None
            self.spline = NaturalCubicSplineND(2, point_set)
        # Periodische Kubische Spline
        elif self.spline_type == SplineType.CUBIC_SPLINE and self.periodic.get():
            assert point_set is not None
            self.spline = PeriodicCubicSplineND(2, point_set)
        # Catmull-Rom-Spline (periodisch und nicht-periodisch)
        elif self.spline_type == SplineType.CATMULL_ROM_SPLINE:
            assert point_set is not None
            self.spline = CatmullRomSplineND(2, point_set, periodic=self.periodic.get())
        # Clamped B-Spline
        elif self.spline_type == SplineType.B_SPLINE and not self.periodic.get():
            n = self.node_count - 3
            knot_vector = dict(enumerate(self.t_nodes))
            knot_vector[-3]  = knot_vector[-2]  = knot_vector[-1]  = knot_vector[0]
            knot_vector[n+1] = knot_vector[n+2] = knot_vector[n+3] = knot_vector[n]
            weights = {i: np.array([self.x_nodes[i+3], self.y_nodes[i+3]]) for i in range(-3, n)}
            self.spline = ClampedBSplineND(2, n, 3, knot_vector, weights)
        # Periodische B-Spline
        elif self.spline_type == SplineType.B_SPLINE and self.periodic.get():
            self.spline = PeriodicBSplineND(2, self.node_count, 3, self.t_nodes, 
                                            np.roll(np.transpose(np.array([self.x_nodes, self.y_nodes])), -2, axis=0))
        # Sonst => Fehler
        else: raise AssertionError("Keine valide Spline ausgewählt!")
            
        assert self.spline is not None
        self.t_graph = np.linspace(self.t_min, self.t_max, self.DENSITY)
        image = self.spline.eval_array(self.t_graph)
        self.x_graph = image[0, :]
        self.y_graph = image[1, :]

        self.update_car()

    def spawn_car(self):
        self.t_car = 0.0
        self.update_car()

    def update_car(self):
        if self.spline is None or self.mode == Mode.DRAW:
            self.show_car.set(False)
            self.run_car.set(False)
        elif self.show_car.get() and self.run_car.get():
            self.t_car = (self.t_car + self.speed.get()) % self.t_max
        if self.show_car.get() and self.spline is not None:
            self.car_pos = np.array([self.spline.eval(self.t_car, deriv=k) for k in range(3)])
        self.update_plot()

    def update_plot(self, event=None):
        self.main_axes.clear()
        self.time_axes.clear()

        self.main_axes.axis('scaled')
        self.main_axes.set_xlim(self.x_min, self.x_max)
        self.main_axes.set_ylim(self.y_min, self.y_max)
        self.main_axes.set_xticks([])
        self.main_axes.set_yticks([])

        # Time-Achse zeichnen
        if self.node_count >= self.get_min_node_count():
            self.time_axes.set_visible(True)
            self.time_axes.set_xlim(self.t_min - 0.01 * (self.t_max - self.t_min),
                                    self.t_max + 0.01 * (self.t_max - self.t_min))
            self.time_axes.set_ylim(-10, 10)
            self.time_axes.set_xticks([])
            self.time_axes.set_yticks([])
            for spine in self.time_axes.spines.values():
                spine.set_visible(False)
            self.time_axes.plot([self.t_min, self.t_max], [0, 0], color='black')
            self.time_axes.scatter(self.t_nodes, np.zeros(self.get_t_node_count()), 
                                   marker=matplotlib.markers.MarkerStyle('|'), 
                                   color='black')
            for j in range(self.get_t_node_count()):
                annotations_t = [fr"$t_{{{j}}}$" for j in range(self.get_t_node_count())]
                self.time_axes.annotate(annotations_t[j], 
                                        (self.t_nodes[j], -2), 
                                        horizontalalignment='center', 
                                        verticalalignment='top')
        else: self.time_axes.set_visible(False)
        
        # Knoten zeichnen
        self.main_axes.scatter(self.x_nodes, self.y_nodes, s=30, 
                               edgecolors='black', facecolors='white', zorder=2.05)
        if self.spline_type == SplineType.B_SPLINE and self.spline != None:
            real_nodes = self.spline.eval_array(self.t_nodes)
            self.main_axes.scatter(real_nodes[0], real_nodes[1], s=20, edgecolors='green', facecolors='white', zorder = 2.00)
        
        # Spline zeichnen
        self.main_axes.plot(self.x_graph, self.y_graph, color='green', zorder=1)

        # Annotationen
        annotations_xy = [fr"$P_{{{j}}}$" for j in range(self.node_count)]
        if self.periodic.get(): annotations_xy[0] = fr"$P_0 = P_{{{self.node_count}}}$"
        for j in range(self.node_count):
            self.main_axes.annotate(annotations_xy[j], 
                                    (self.x_nodes[j], self.y_nodes[j]), 
                                    horizontalalignment='right', 
                                    verticalalignment='top', zorder=2.1)
            
        # Auto zeichnen
        if self.show_car.get():
            # Punkte in main und time
            angle = np.arctan2(self.car_pos[1,1], self.car_pos[1,0])
            points = np.array([[np.cos(angle + 2*k*np.pi/3), np.sin(angle + 2*k*np.pi/3)] for k in range(4)])
            custom_marker = matplotlib.markers.MarkerStyle(matplotlib.path.Path(points))
            self.main_axes.scatter(self.car_pos[0,0], self.car_pos[0,1], s=200, 
                                   edgecolors='black', facecolors='#bb5bff', zorder=2.5,
                                   marker=custom_marker)
            self.time_axes.scatter(self.t_car, 0, s=60,
                                   edgecolors='black', facecolors='#bb5bff', zorder=2.5)
            if self.show_derivs.get():
                # v- und a-Vektoren
                self.main_axes.arrow(self.car_pos[0,0], self.car_pos[0,1], 
                                     self.car_pos[1,0] * self.ARROW_SCALE, self.car_pos[1,1] * self.ARROW_SCALE, 
                                     length_includes_head=True, head_width=0.01,
                                     color='blue', zorder=2.3)
                self.main_axes.arrow(self.car_pos[0,0], self.car_pos[0,1], 
                                     self.car_pos[2,0] * self.ARROW_SCALE, self.car_pos[2,1] * self.ARROW_SCALE, 
                                     length_includes_head=True, head_width=0.01,
                                     color='red', zorder=2.3)

        self.canvas.draw_idle()

    def trigger_drawing_spline(self, *args):
        if self.mode == Mode.NONE:
            self.reset_points()
            self.mode = Mode.DRAW
            self.draw_button.config(text="Fertig")
        elif self.mode == Mode.DRAW:
            self.mode = Mode.NONE
            self.draw_button.config(text="Neu Zeichnen")
        else: return



    def get_close_node_index(self, mouse_x, mouse_y) -> int | None:
        for j in range(self.node_count):
            if abs(mouse_x - self.x_nodes[j]) <= 0.5 * self.EXCLUSION_DIST_XY and abs(mouse_y - self.y_nodes[j]) <= 0.5 * self.EXCLUSION_DIST_XY:
                return j
        return None
    
    def get_close_time_index(self, mouse_x, mouse_y) -> int | None:
        for j in range(self.get_t_node_count()):
            if abs(mouse_x - self.t_nodes[j]) <= 0.5 * self.EXCLUSION_RATIO_T * self.get_t_node_count():
                return j
        return None
    
    def on_press(self, event: MouseEvent): 
        # Invalide Event-Daten => Abbruch
        if event.xdata is None or event.ydata is None: return
        # Maus im Main-Canvas
        elif self.mode == Mode.NONE and event.inaxes == self.main_axes:
            # Auto bewegen
            if not self.run_car.get() and abs(event.xdata - self.car_pos[0,0]) <= 0.5 * self.EXCLUSION_DIST_XY and abs(event.ydata - self.car_pos[0,1]) <= 0.5 * self.EXCLUSION_DIST_XY:
                self.mode = Mode.CAR_XY
            # Punkt bewegen
            elif (j := self.get_close_node_index(event.xdata, event.ydata)) is not None:
                self.mode = Mode.MOVE_XY
                self.dragging_index = j
        # Maus im Time-Canvas
        elif self.mode == Mode.NONE and event.inaxes == self.time_axes:
            # Auto bewegen
            if not self.run_car.get() and abs(event.xdata - self.t_car) <= 0.5 * self.EXCLUSION_RATIO_T * self.get_t_node_count():
                self.mode = Mode.CAR_T
            # Zeitpunkt bewegen
            elif (j := self.get_close_time_index(event.xdata, event.ydata)) not in (None, 0, self.get_t_node_count()-1):
                self.mode = Mode.MOVE_T
                self.dragging_index = j
        # Spline zeichnen
        elif self.mode == Mode.DRAW and event.inaxes == self.main_axes:
            cond_1 = np.array([abs(self.x_nodes[i] - event.xdata) >= self.EXCLUSION_DIST_XY for i in range(self.node_count)])
            cond_2 = np.array([abs(self.y_nodes[i] - event.ydata) >= self.EXCLUSION_DIST_XY for i in range(self.node_count)])
            if self.node_count == 0 or all(cond_1 | cond_2):
                self.node_count += 1
                if self.spline_type == SplineType.B_SPLINE:
                    if self.periodic.get(): raise AssertionError("Periodisch kann nicht gezeichnet werden")
                    else:
                        if self.node_count < self.get_min_node_count():
                            self.t_max = 1
                            self.t_nodes = np.array([0.])
                        else: 
                            self.t_max = self.get_t_node_count() - 1
                            self.t_nodes = np.append(self.t_nodes, self.t_max)
                else:
                    self.t_max = self.node_count - 1
                    self.t_nodes = np.append(self.t_nodes, self.t_max)
                self.x_nodes = np.append(self.x_nodes, event.xdata)
                self.y_nodes = np.append(self.y_nodes, event.ydata)
                self.update_spline()

    def on_release(self, event: MouseEvent):
        if self.mode in (Mode.MOVE_XY, Mode.MOVE_T, Mode.CAR_XY, Mode.CAR_T):
            self.mode = Mode.NONE
            self.dragging_index = None

    def on_motion(self, event: MouseEvent):
        # Invalide Daten
        if event.xdata is None or event.ydata is None: return
        # Punke bewegen
        elif self.mode == Mode.MOVE_XY and event.inaxes == self.main_axes:
            assert self.dragging_index is not None
            j = self.dragging_index
            cond_1 = np.array([i == j or abs(self.x_nodes[i] - event.xdata) >= self.EXCLUSION_DIST_XY for i in range(self.node_count)])
            cond_2 = np.array([i == j or abs(self.y_nodes[i] - event.ydata) >= self.EXCLUSION_DIST_XY for i in range(self.node_count)])
            if all(cond_1 | cond_2):
                self.x_nodes[j] = restrict_to_interval(event.xdata, self.x_min, self.x_max)
                self.y_nodes[j] = restrict_to_interval(event.ydata, self.y_min, self.y_max)
            self.update_spline()
        # Zeit bewegen
        elif self.mode == Mode.MOVE_T and event.inaxes == self.time_axes:
            assert self.dragging_index is not None
            j = self.dragging_index
            if all(i == j or abs(self.t_nodes[i] - event.xdata) >= self.EXCLUSION_RATIO_T * self.get_t_node_count() for i in range(self.get_t_node_count())):
                self.t_nodes[j] = restrict_to_interval(event.xdata, self.t_min, self.t_max)
                argsort = np.argsort(self.t_nodes, kind='mergesort')
                self.dragging_index = argsort[j]
                self.t_nodes = self.t_nodes[argsort]
            self.update_spline()
        # Auto in Main bewegen
        elif self.mode == Mode.CAR_XY and event.inaxes == self.main_axes and not self.run_car.get():
            points_on_spline = np.sum((np.transpose(np.array([self.x_graph, self.y_graph])) - np.array([event.xdata, event.ydata]))**2, axis=1)
            closest_index = np.argmin(points_on_spline)
            self.t_car = restrict_to_interval(self.t_graph[closest_index], self.t_min, self.t_max)
            self.update_car()
        # Auto in Time bewegen
        elif self.mode == Mode.CAR_T and event.inaxes == self.time_axes and not self.run_car.get():
            self.t_car = restrict_to_interval(event.xdata, self.t_min, self.t_max)
            self.update_car()

    def repeat(self):
        if self.show_car.get() and self.run_car.get() and self.spline is not None:
            self.update_car()
        self.root.after(1000 // self.FRAME_RATE, self.repeat)






if __name__ == '__main__':
    root = tk.Tk()
    app = SplineInterpolatingPointsApp(root)
    def on_close():
        root.quit()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', on_close)
    system = root.tk.call("tk", "windowingsystem")
    if system == "win32":
        root.state("zoomed")
    elif system == "x11":
        root.attributes("-zoomed", True)
    ttk.Style().theme_use('alt')
    root.mainloop()


