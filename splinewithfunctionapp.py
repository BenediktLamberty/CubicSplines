from splineclasses import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec 
import matplotlib.figure 
import matplotlib.axes
import sympy


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

def safely_eval_array(f: Callable[[np.ndarray], np.ndarray], t: np.ndarray) -> np.ndarray:
    with np.errstate(all='ignore'):
        result = f(t)
    return np.nan_to_num(result, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


class SplineInterpolatingPointsApp:
    root: tk.Tk
    x_coordinates:  np.ndarray
    y_coordinates:  np.ndarray
    func_string:    tk.StringVar
    dist_string:    tk.StringVar
    func_lambdas:   list[sympy.FunctionClass]
    node_count:     tk.IntVar
    left_border:    tk.DoubleVar
    right_border:   tk.DoubleVar
    show_derivs:    tk.BooleanVar
    draw_function:  tk.BooleanVar
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
        self.root.title("Kubische Splines -- Approximation von Funktionen")
        self.root.geometry("1000x600")

        #  Variablen
        self.node_count     = tk.IntVar(value=4)
        self.left_border    = tk.DoubleVar(value=0.0)
        self.right_border   = tk.DoubleVar(value=1.0)
        self.show_derivs    = tk.BooleanVar(value=True)
        self.draw_function  = tk.BooleanVar(value=True)
        self.draw_poly      = tk.BooleanVar(value=False)
        self.draw_linear    = tk.BooleanVar(value=False)
        self.draw_quadratic = tk.BooleanVar(value=False)
        self.draw_natural   = tk.BooleanVar(value=False)
        self.draw_periodic  = tk.BooleanVar(value=False)
        self.draw_clamped   = tk.BooleanVar(value=False)
        self.func_string    = tk.StringVar(value="0")
        self.dist_string    = tk.StringVar(value="")
        sym_x = sympy.symbols("x")
        self.func_lambdas = [sympy.lambdify(sym_x, 0.0, "numpy")] * 4  # For some reason always returns 0??????? TODO

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
        ttk.Button(node_count_row, text="--", width=3, command=lambda: self.change_node_count(-10)).pack(side=tk.LEFT)        
        ttk.Button(node_count_row, text="-", width=3, command=lambda: self.change_node_count(-1)).pack(side=tk.LEFT)
        ttk.Button(node_count_row, text="+", width=3, command=lambda: self.change_node_count(+1)).pack(side=tk.LEFT)
        ttk.Button(node_count_row, text="++", width=3, command=lambda: self.change_node_count(+10)).pack(side=tk.LEFT)
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Funktion:").pack(anchor=tk.W)
        func_row = ttk.Frame(control_frame)
        func_row.pack(anchor=tk.W)
        ttk.Label(func_row, text="f(x) =").pack(side=tk.LEFT)
        ttk.Entry(func_row, textvariable=self.func_string, width=20).pack(side=tk.LEFT)
        ttk.Button(func_row, text="✓", width=3, command=self.reset_function).pack(side=tk.LEFT)
        ttk.Separator(control_frame).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Intervallgrenzen:").pack(anchor=tk.W)
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
        ttk.Checkbutton(control_frame, text="Funktion anzeigen",
                        variable=self.draw_function,
                        command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Ableitungen anzeigen",
                        variable=self.show_derivs,
                        command=self.update_plot).pack(anchor=tk.W)
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

        self.reset_points()

    def reset_variables(self):
        self.node_count.set(value=4)
        self.left_border.set(value=0.0)
        self.right_border.set(value=1.0)
        self.show_derivs.set(value=True)
        self.draw_function.set(value=True)
        self.draw_poly.set(value=False)
        self.draw_linear.set(value=False)
        self.draw_quadratic.set(value=False)
        self.draw_natural.set(value=False)
        self.draw_periodic.set(value=False)
        self.draw_clamped.set(value=False)
        self.func_string.set(value="0")

    def eval_array_function(self, t: np.ndarray, deriv: int) -> np.ndarray:
        result = safely_eval_array(self.func_lambdas[deriv], t)
        if isinstance(result, np.ndarray): return result
        else: return np.zeros(len(t)) + result

    def reset_everything(self):
        self.reset_variables()
        self.reset_function()
        self.reset_points()

    def get_bounds(self) -> tuple[float, float]:
        a = safely_get_float(self.left_border)
        b = safely_get_float(self.right_border)
        a, b = sort_unequal_pair(a, b)
        return a, b
    
    def check_periodicity(self):
        if abs(self.y_coordinates[0] - self.y_coordinates[-1]) >= 1e-6: 
            self.draw_periodic.set(False)
        self.update_plot()

    def enforce_periodicity(self):
        self.y_coordinates[-1] = self.y_coordinates[0]
        self.update_plot()
    
    def get_node_count(self) -> int:
        MAX_NODE_COUNT = 100
        n = safely_get_int(self.node_count)
        if n < 1: return 1
        elif 1 <= n <= MAX_NODE_COUNT: return n
        else: return MAX_NODE_COUNT

    def change_node_count(self, change: int):
        n = self.get_node_count()
        n += change
        if n < 1: n = 1
        elif n > 100: n = 100
        self.node_count.set(n)
        self.reset_points()

    def reset_function(self, *args):
        a, b = self.get_bounds()
        string: str = self.func_string.get()
        if string == "": return
        string = string.replace("^", "**").replace("a", f"{a}").replace("b", f"{b}").replace("²", "**2").replace("³", "**3").replace("e", "2.71828182845904")
        lambda_float = lambda x: 0
        try:
            # lambda_float = eval("lambda x: " + string, {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi, "e": np.e})
            # x = symbols('x')
            # expr = parse_expr("x + x + 2*x", local_dict={"x": x})
            x = sympy.symbols("x")
            exprs: list[sympy.Expr] = []
            exprs.append(sympy.sympify(string, locals={"x": x}) + 0*x)
            for k in range(1, 4):
                exprs.append(sympy.diff(exprs[0], x, k))
            for k in range(4):
                self.func_lambdas[k] = sympy.lambdify(x, exprs[k], "numpy")
            # self.func_lambda_float = lambda_float
            # self.func_lambda_array = lambda x: 0 * x + lambda_float(x)
            # self.func_expr = sym_expr
            self.reset_points()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid equation:\n{e}")


    def reset_points(self, *args):
        n = self.get_node_count()
        a, b = self.get_bounds()
        self.x_coordinates = np.linspace(a, b, n+1)
        self.y_coordinates = self.eval_array_function(self.x_coordinates, 0) # TODO Exceptions
        self.update_plot()  

    def plot_curve_with_derivs(self, curve: InterpolatingCurve1D, interval: np.ndarray, color: str):
        for k in range(4):
            graph = curve.eval_array(interval, deriv=k)
            self.axes[k].plot(interval, graph, color=color)
    
    def update_plot(self, event=None):
        n = self.get_node_count()
        a, b = self.get_bounds()

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
            self.axes[k].set_xticks(self.x_coordinates)
            self.axes[k].set_xticklabels([])
            # self.axes[k].set_yticks([y_min, 0, y_max] if y_min < 0 < y_max else [y_min, y_max])
            # self.axes[k].set_yticklabels([])

        # self.axes[0].set_ylim(y_min, y_max)
        # self.axes[0].set_yticklabels([])
        labels = []
        if n <= 9:
            labels = [rf"$x_{j} = {self.x_coordinates[j]:.2f}$" for j in range(self.get_node_count() + 1)]
        elif n <= 20: 
            labels = [rf"${self.x_coordinates[j]:.2f}$" for j in range(self.get_node_count() + 1)]
        elif n <= 40: 
            labels = [rf"${self.x_coordinates[j]:.2f}$" if j%2==0 else "" for j in range(self.get_node_count() + 1)]
        else:
            labels = [rf"${self.x_coordinates[j]:.2f}$" if j%4==0 else "" for j in range(self.get_node_count() + 1)]
        self.axes[0].set_xticklabels(labels)
        # self.axes[0].set_yticklabels([rf"${y_min}$", rf"${0}$", rf"${y_max}$"] if y_min < 0 < y_max else [rf"${y_min}$", rf"${y_max}$"])

        self.axes[1].set_title("1. Ableitung")
        self.axes[2].set_title("2. Ableitung")
        self.axes[3].set_title("3. Ableitung")

        # Splines Zeichnen
        DENSITY = 500
        interval = np.linspace(a, b, DENSITY)
        point_set = PointSet1D(list(self.x_coordinates), list(self.y_coordinates))

        if self.draw_function.get():
            for k in range(4):
                self.axes[k].plot(interval, self.eval_array_function(interval, k), color="#B1B1B1")
        if self.draw_poly.get():
            self.plot_curve_with_derivs(InterpolatingPolynomial1D(point_set), interval, "#de0000")
        if self.draw_linear.get():
            self.plot_curve_with_derivs(LinearSpline1D(point_set), interval, "#fa6f05")
        if self.draw_quadratic.get():
            self.plot_curve_with_derivs(QuadraticSpline1D(point_set, 0), interval, "#fad105")  # TODO
        if self.draw_natural.get():
            self.plot_curve_with_derivs(NaturalCubicSpline1D(point_set), interval, "green")
        if self.draw_periodic.get() and abs(self.y_coordinates[0] - self.y_coordinates[-1]) < 1e-6:
            self.plot_curve_with_derivs(PeriodicCubicSpline1D(point_set), interval, "blue")
        if self.draw_clamped.get():
            self.plot_curve_with_derivs(ClampedCubicSpline1D(point_set, tuple(self.eval_array_function(np.array([a, b]), 1))), interval, "#ae0ff7")

        self.axes[0].scatter(self.x_coordinates, self.y_coordinates, s=30, edgecolors="black", facecolors="white", zorder=2)
        
        self.canvas.draw_idle()





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


