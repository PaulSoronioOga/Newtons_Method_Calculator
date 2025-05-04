# Newtons_Method_Calculator
# Newton's Method for Optimization Calculator

This is an interactive **Streamlit-based calculator** for finding the **maximum** or **minimum** of a function using **Newton's Method for Optimization**. It provides symbolic differentiation, step-by-step iteration details, convergence checking, and visualization through plots.

---

## What is Newton's Method for Optimization?

Newton's Method is an iterative algorithm used to find **critical points** (where the first derivative is zero) of a function to determine **local maxima** or **minima**.

It uses the update rule:

\[
x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}
\]

Where:
- \( f'(x) \) is the first derivative (slope),
- \( f''(x) \) is the second derivative (curvature).

If:
- \( f''(x) > 0 \), the point is a **minimum**.
- \( f''(x) < 0 \), the point is a **maximum**.

---

## How to Run the Calculator

### Requirements:
- Python 3.8+
- Streamlit
- SymPy
- NumPy
- Matplotlib
- Pandas

### Installation:
```bash
pip install streamlit sympy numpy matplotlib pandas
