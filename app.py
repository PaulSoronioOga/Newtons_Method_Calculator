import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Title and subtitle ---
st.markdown("<h1 style='font-family:sans-serif;'>Newton's Method for Optimization</h1>", unsafe_allow_html=True)
st.write("This calculator is used to solve the maxima and minima of functions with the use of Newton's Method.")

# --- Inputs ---
x = sp.symbols('x')
func_input = st.text_input("Enter the function", "x**2 - 4*x + 5")

# --- Math display of input ---
try:
    f_expr = sp.sympify(func_input)
    st.latex(f"f(x) = {sp.latex(f_expr)}")
except Exception:
    st.error("Invalid function.")
    st.stop()

x0 = st.number_input("Enter the initial guess (x₀):", value=0.0)
tol_input = st.text_input("Enter number of tolerance", "1e-6")
try:
    tol = float(tol_input)
except ValueError:
    st.error("Invalid tolerance. Use formats like 1e-6 or 0.0001")
    st.stop()

max_iter = st.number_input("Enter number of iterations", min_value=1, value=20)
goal = st.selectbox("What to find?", ["Minimum", "Maximum"])
show_option = st.radio("Show the following:", ["Graph", "Iteration table"])

# --- Derivatives ---
f_prime_expr = sp.diff(f_expr, x)
f_double_prime_expr = sp.diff(f_prime_expr, x)
st.markdown("### 📐 Derivatives:")
st.latex(f"f'(x) = {sp.latex(f_prime_expr)}")
st.latex(f"f''(x) = {sp.latex(f_double_prime_expr)}")

# --- Numerical functions ---
f = sp.lambdify(x, f_expr, "numpy")
f_prime = sp.lambdify(x, f_prime_expr, "numpy")
f_double_prime = sp.lambdify(x, f_double_prime_expr, "numpy")

# --- Newton's Method ---
st.markdown("### 🔁 Step by Step Solution:")
xn = x0
iteration_data = []

for i in range(int(max_iter)):
    f1 = f_prime(xn)
    f2 = f_double_prime(xn)
    if f2 == 0:
        st.error(f"Iteration {i+1}: Division by zero.")
        break

    step = f1 / f2
    xn_next = xn - step
    error = abs(xn_next - xn)

    iteration_data.append({
        "Iteration": i + 1,
        "xₙ": xn,
        "f'(xₙ)": f1,
        "f''(xₙ)": f2,
        "xₙ₊₁": xn_next,
        "Error": error
    })

    # Show LaTeX step
    st.markdown(f"**Iteration {i+1}:**")
    st.latex(rf"x_{i} = {xn:.6f}")
    st.latex(rf"f'(x_{i}) = {f1:.6f}, \quad f''(x_{i}) = {f2:.6f}")
    st.latex(rf"x_{{{i+1}}} = x_{i} - \frac{{f'(x_{i})}}{{f''(x_{i})}} = {xn:.6f} - \frac{{{f1:.6f}}}{{{f2:.6f}}} = {xn_next:.6f}")
    st.latex(rf"\text{{Error}} = |x_{{{i+1}}} - x_{i}| = {error:.2e}")

    if error < tol:
        curvature = f_double_prime(xn_next)
        kind = "minimum" if curvature > 0 else "maximum" if curvature < 0 else "saddle point"
        if (goal == "Minimum" and curvature > 0) or (goal == "Maximum" and curvature < 0):
            st.success(f"✅ Local {kind} found at x = {xn_next:.6f}")
        else:
            st.warning(f"⚠️ Stationary point at x = {xn_next:.6f} is not a local {goal.lower()}.")
        break

    xn = xn_next
else:
    st.warning("Did not converge within the maximum number of iterations.")

# --- Plot or table ---
if show_option == "Graph":
    x_vals = np.linspace(x0 - 10, x0 + 10, 400)
    y_vals = f(x_vals)

    # Iteration points
    x_points = [data["xₙ"] for data in iteration_data]
    y_points = [f(xi) for xi in x_points]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.plot(x_points, y_points, 'ro-', label='Convergence path')
    plt.scatter([x0], [f(x0)], color='orange', s=80, label='Initial guess')
    plt.scatter([x_points[-1]], [y_points[-1]], color='green', s=100, label='Final point')

    for i in range(len(x_points) - 1):
        plt.annotate('', xy=(x_points[i+1], y_points[i+1]), xytext=(x_points[i], y_points[i]),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.title("Newton's Method Convergence")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
else:
    df = pd.DataFrame(iteration_data)
    st.dataframe(df.style.format({
        "xₙ": "{:.6f}",
        "f'(xₙ)": "{:.6f}",
        "f''(xₙ)": "{:.6f}",
        "xₙ₊₁": "{:.6f}",
        "Error": "{:.2e}"
    }), use_container_width=True)
