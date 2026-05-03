from splineclasses import *
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseEvent
import matplotlib.gridspec 
import matplotlib.figure 
import matplotlib.axes


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


class SplineInterpolatingPointsApp:
    root: tk.Tk
    x_coordinates:  list[float]
    y_coordinates:  list[float]
    node_count:     tk.IntVar
    left_border:    tk.DoubleVar
    right_border:   tk.DoubleVar
    top_border:     tk.DoubleVar
    bottom_border:  tk.DoubleVar
    clamp_a:        tk.DoubleVar
    clamp_b:        tk.DoubleVar
    moveable_x:     tk.BooleanVar
    moveable_y:     tk.BooleanVar
    lock_period:    tk.BooleanVar
    show_derivs:    tk.BooleanVar
    draw_poly:      tk.BooleanVar
    draw_linear:    tk.BooleanVar
    draw_quadratic: tk.BooleanVar
    draw_natural:   tk.BooleanVar
    draw_periodic:  tk.BooleanVar
    draw_clamped:   tk.BooleanVar
    warning:  str = ""
    canvas:   FigureCanvasTkAgg
    figure:   matplotlib.figure.Figure
    gridspec: matplotlib.gridspec.GridSpec
    axes:     list[matplotlib.axes.Axes]
    dragging_index: int | None = None
    
    def __init__(self, root: tk.Tk):
        # Root
        self.root = root
        self.root.title("Kubische Splines -- Inerpolation von Werten")
        self.root.geometry("1000x600")

        #  Variablen
        self.node_count     = tk.IntVar(value=4)
        self.left_border    = tk.DoubleVar(value=0.0)
        self.right_border   = tk.DoubleVar(value=1.0)
        self.top_border     = tk.DoubleVar(value=1.0)
        self.bottom_border  = tk.DoubleVar(value=-1.0)
        self.clamp_a        = tk.DoubleVar(value=0.0)
        self.clamp_b        = tk.DoubleVar(value=0.0)
        self.moveable_x     = tk.BooleanVar(value=True)
        self.moveable_y     = tk.BooleanVar(value=True)
        self.lock_period    = tk.BooleanVar(value=False)
        self.show_derivs    = tk.BooleanVar(value=True)
        self.draw_poly      = tk.BooleanVar(value=False)
        self.draw_linear    = tk.BooleanVar(value=False)
        self.draw_quadratic = tk.BooleanVar(value=False)
        self.draw_natural   = tk.BooleanVar(value=False)
        self.draw_periodic  = tk.BooleanVar(value=False)
        self.draw_clamped   = tk.BooleanVar(value=False)

        # Frames
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inhalte vom Control-Frame
        ttk.Label(control_frame, text="Anzahl an Abschnitten:").pack(anchor=tk.W)
        node_count_row = ttk.Frame(control_frame)
        node_count_row.pack(anchor=tk.W)
        ttk.Label(node_count_row, text="n =").pack(side=tk.LEFT)
        node_count_entry = ttk.Entry(node_count_row, textvariable=self.node_count, width=5)
        node_count_entry.pack(side=tk.LEFT)
        node_count_entry.bind("<KeyRelease>", lambda e: self.reset_points())
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Achsen-Skalierung:").pack(anchor=tk.W)
        x_axis_row = ttk.Frame(control_frame)
        x_axis_row.pack(anchor=tk.CENTER)
        ttk.Label(x_axis_row, text="a =").pack(side=tk.LEFT)
        left_border_entry = ttk.Entry(x_axis_row, textvariable=self.left_border, width=5)
        left_border_entry.pack(side=tk.LEFT)
        left_border_entry.bind("<KeyRelease>", lambda e: self.reset_points())
        ttk.Label(x_axis_row, text="≤ x ≤").pack(side=tk.LEFT)
        right_border_entry = ttk.Entry(x_axis_row, textvariable=self.right_border, width=5)
        right_border_entry.pack(side=tk.LEFT)
        right_border_entry.bind("<KeyRelease>", lambda e: self.reset_points())
        ttk.Label(x_axis_row, text="= b").pack(side=tk.LEFT)
        y_axis_row = ttk.Frame(control_frame)
        y_axis_row.pack(anchor=tk.CENTER)
        bottom_border_entry = ttk.Entry(y_axis_row, textvariable=self.bottom_border, width=5)
        bottom_border_entry.pack(side=tk.LEFT)
        bottom_border_entry.bind("<KeyRelease>", lambda e: self.reset_points())
        ttk.Label(y_axis_row, text="≤ y ≤").pack(side=tk.LEFT)
        top_border_entry = ttk.Entry(y_axis_row, textvariable=self.top_border, width=5)
        top_border_entry.pack(side=tk.LEFT)
        top_border_entry.bind("<KeyRelease>", lambda e: self.reset_points())
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Randbedingungen:").pack(anchor=tk.W)
        clamp_row = ttk.Frame(control_frame)
        clamp_row.pack(anchor=tk.W)
        ttk.Label(clamp_row, text="S'(a) =").pack(side=tk.LEFT)
        clamp_a_entry = ttk.Entry(clamp_row, textvariable=self.clamp_a, width=5)
        clamp_a_entry.pack(side=tk.LEFT)
        clamp_a_entry.bind("<KeyRelease>", lambda e: self.update_plot())
        ttk.Label(clamp_row, text="  S'(b) =").pack(side=tk.LEFT)
        clamp_b_entry = ttk.Entry(clamp_row, textvariable=self.clamp_b, width=5)
        clamp_b_entry.pack(side=tk.LEFT)
        clamp_b_entry.bind("<KeyRelease>", lambda e: self.update_plot())
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        checkbutton_lambda = lambda: self.update_plot() 
        ttk.Label(control_frame, text="Interpolationsart:").pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Polynom",
                        variable=self.draw_poly,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="lineare Spline",
                        variable=self.draw_linear,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="quadratische Spline",
                        variable=self.draw_quadratic,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="natürliche kubische Spline",
                        variable=self.draw_natural,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="periodische kubische Spline",
                        variable=self.draw_periodic,
                        command=self.check_periodicity).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="eingespannte kubische Spline",
                        variable=self.draw_clamped,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Einstellungen:").pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Punkte in x-Richtung verschiebbar",
                        variable=self.moveable_x,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Punkte in y-Richtung verschiebbar",
                        variable=self.moveable_y,
                        command=checkbutton_lambda).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Ableitungen anzeigen",
                        variable=self.show_derivs,
                        command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Periodizität erzwingen",
                        variable=self.lock_period,
                        command=self.enforce_periodicity).pack(anchor=tk.W)
        ttk.Button(control_frame, text="Alles Zurücksetzen", command=self.reset_everything).pack(anchor=tk.W)

        
        # Canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.gridspec = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[3, 1])
        self.axes = []
        self.axes.append(self.figure.add_subplot(self.gridspec[0, :]))
        self.axes.append(self.figure.add_subplot(self.gridspec[1, 0]))
        self.axes.append(self.figure.add_subplot(self.gridspec[1, 1]))
        self.axes.append(self.figure.add_subplot(self.gridspec[1, 2]))

        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.reset_points()

    def reset_variables(self):
        self.node_count.set(value=4)
        self.left_border.set(value=0.0)
        self.right_border.set(value=1.0)
        self.top_border.set(value=1.0)
        self.bottom_border.set(value=-1.0)
        self.clamp_a.set(value=0.0)
        self.clamp_b.set(value=0.0)
        self.moveable_x.set(value=True)
        self.moveable_y.set(value=True)
        self.lock_period.set(value=False)
        self.show_derivs.set(value=True)
        self.draw_poly.set(value=False)
        self.draw_linear.set(value=False)
        self.draw_quadratic.set(value=False)
        self.draw_natural.set(value=False)
        self.draw_periodic.set(value=False)
        self.draw_clamped.set(value=False)

    def reset_everything(self):
        self.reset_variables()
        self.reset_points()

    def get_bounds(self) -> tuple[float, float, float, float]:
        a = safely_get_float(self.left_border)
        b = safely_get_float(self.right_border)
        a, b = sort_unequal_pair(a, b)
        y_min = safely_get_float(self.bottom_border)
        y_max = safely_get_float(self.top_border)
        y_min, y_max = sort_unequal_pair(y_min, y_max)
        return a, b, y_min, y_max
    
    def check_periodicity(self):
        if abs(self.y_coordinates[0] - self.y_coordinates[-1]) >= 1e-6: 
            self.draw_periodic.set(False)
        self.update_plot()

    def enforce_periodicity(self):
        self.y_coordinates[-1] = self.y_coordinates[0]
        self.update_plot()
    
    def get_node_count(self) -> int:
        n = safely_get_int(self.node_count)
        if n < 1: return 1
        elif 1 <= n <= 9: return n
        else: return 9

    def reset_points(self):
        n = self.get_node_count()
        a = safely_get_float(self.left_border)
        b = safely_get_float(self.right_border)
        a, b = sort_unequal_pair(a, b)
        self.x_coordinates = list(np.linspace(a, b, n+1))
        self.y_coordinates = list(np.zeros(n+1))
        self.update_plot()  


    def plot_curve_with_derivs(self, curve: InterpolatingCurve1D, interval: np.ndarray, color: str):
        for k in range(4):
            graph = curve.eval_array(interval, deriv=k)
            self.axes[k].plot(interval, graph, color=color)
    
    def update_plot(self, event=None):
        sorted_x, sorted_y = sort_points_by_x(self.x_coordinates, self.y_coordinates)
        n = self.get_node_count()
        a, b, y_min, y_max = self.get_bounds()

        for k in range(4):
            self.axes[k].clear()

        for k in range(1, 4):
            self.axes[k].set_visible(self.show_derivs.get())

        # Style Polots
        for k in range(4):
            self.axes[k].tick_params(
                axis='both', which='both',
                top=True, bottom=True,
                left=True, right=True)
            self.axes[k].set_xlim(a, b)
            # self.axes[k].set_ylim(y_min, y_max)
            self.axes[k].set_xticks(sorted_x)
            self.axes[k].set_xticklabels([])
            # self.axes[k].set_yticks([y_min, 0, y_max] if y_min < 0 < y_max else [y_min, y_max])
            # self.axes[k].set_yticklabels([])

        self.axes[0].set_ylim(y_min, y_max)
        # self.axes[0].set_yticklabels([])
        labels = [rf"$x_{j} = {sorted_x[j]:.2f}$" for j in range(self.get_node_count() + 1)]
        self.axes[0].set_xticklabels(labels)
        # self.axes[0].set_yticklabels([rf"${y_min}$", rf"${0}$", rf"${y_max}$"] if y_min < 0 < y_max else [rf"${y_min}$", rf"${y_max}$"])

        self.axes[1].set_title("1. Ableitung")
        self.axes[2].set_title("2. Ableitung")
        self.axes[3].set_title("3. Ableitung")

        # Splines Zeichnen
        DENSITY = 500
        interval = np.linspace(a, b, DENSITY)
        point_set = PointSet1D(sorted_x, sorted_y)

        if self.draw_poly.get():
            self.plot_curve_with_derivs(InterpolatingPolynomial1D(point_set), interval, "#de0000")
        if self.draw_linear.get():
            self.plot_curve_with_derivs(LinearSpline1D(point_set), interval, "#fa6f05")
        if self.draw_quadratic.get():
            self.plot_curve_with_derivs(QuadraticSpline1D(point_set, safely_get_float(self.clamp_a)), interval, "#fad105")
        if self.draw_natural.get():
            self.plot_curve_with_derivs(NaturalCubicSpline1D(point_set), interval, "green")
        if self.draw_periodic.get() and abs(sorted_y[0] - sorted_y[-1]) < 1e-6:
            self.plot_curve_with_derivs(PeriodicCubicSpline1D(point_set), interval, "blue")
        if self.draw_clamped.get():
            self.plot_curve_with_derivs(ClampedCubicSpline1D(point_set, (safely_get_float(self.clamp_a), safely_get_float(self.clamp_b))), interval, "#ae0ff7")

        self.axes[0].scatter(sorted_x, sorted_y, s=30, edgecolors="black", facecolors="white", zorder=2)
        
        self.canvas.draw_idle()

    def get_index_of_close_point(self, mouse_x, mouse_y, dist_cutoff_ratio = 0.005) -> int | None:
        n = self.get_node_count()
        a, b, y_min, y_max = self.get_bounds()
        dist_cutoff_x = (b - a) * dist_cutoff_ratio
        dist_cutoff_y = (y_max - y_min) * dist_cutoff_ratio
        for j in range(n+1):
            if abs(mouse_x - self.x_coordinates[j]) <= dist_cutoff_x and abs(mouse_y - self.y_coordinates[j]) <= dist_cutoff_y:
                return j
        return None
    
    def on_press(self, event: MouseEvent): 
        if event.inaxes != self.axes[0]: return
        j = self.get_index_of_close_point(event.xdata, event.ydata)
        if j is not None:
            self.dragging_index = j

    def on_release(self, event: MouseEvent):
        self.dragging_index = None

    def on_motion(self, event: MouseEvent):
        if self.dragging_index is None or event.inaxes != self.axes[0]: return
        n = self.get_node_count()
        a, b, y_min, y_max = self.get_bounds()
        j = self.dragging_index

        EXCLUSION_RATIO = 0.01
        exclusion_radius_x = (b - a) * EXCLUSION_RATIO

        if self.moveable_x.get() and j not in [0, n] and event.xdata is not None and a < event.xdata < b:
            if all(i == j or abs(self.x_coordinates[i] - event.xdata) >= exclusion_radius_x for i in range(n+1)):
                self.x_coordinates[j] = event.xdata

        if self.moveable_y.get() and event.ydata is not None and y_min < event.ydata < y_max:
            self.y_coordinates[j] = event.ydata
            if j == 0 and self.lock_period.get(): self.y_coordinates[n] = event.ydata
            if j == n and self.lock_period.get(): self.y_coordinates[0] = event.ydata

        self.update_plot()





if __name__ == '__main__':
    root = tk.Tk()
    app = SplineInterpolatingPointsApp(root)
    def on_close():
        root.quit()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    system = root.tk.call("tk", "windowingsystem")
    if system == "win32":
        root.state("zoomed")
    elif system == "x11":
        root.attributes("-zoomed", True)
    ttk.Style().theme_use("alt")
    root.mainloop()


