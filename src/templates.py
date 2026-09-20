# src/templates.py

import re


GREEK = {
    'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma',
    'delta': '\\delta', 'epsilon': '\\epsilon', 'theta': '\\theta',
    'lambda': '\\lambda', 'mu': '\\mu', 'pi': '\\pi',
    'sigma': '\\sigma', 'phi': '\\phi', 'omega': '\\omega',
    'rho': '\\rho', 'tau': '\\tau', 'eta': '\\eta', 'kappa': '\\kappa',
    'nu': '\\nu', 'xi': '\\xi', 'zeta': '\\zeta', 'psi': '\\psi',
}

GREEK_INF = {
    'inf': '\\infty', 'infinity': '\\infty', '∞': '\\infty',
    'бесконечность': '\\infty', 'infinity': '\\infty',
}


def _convert_expr(expr: str) -> str:
    """Конвертирует простое выражение в LaTeX (для вложенного использования)."""
    expr = expr.strip()
    
    # a/b^n
    m = re.match(r'^(\w+)/(\w+)\^(\d+)$', expr)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}^{{{m.group(3)}}}}}"
    
    # a/b
    m = re.match(r'^(\w+)/(\w+)$', expr)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    
    # a^n
    m = re.match(r'^([a-zA-Z])\^(\d+)$', expr)
    if m:
        return f"{m.group(1)}^{{{m.group(2)}}}"
    
    # e^(-x^2)
    m = re.match(r'^(\w+)\^\((.+)\)$', expr)
    if m:
        inner = m.group(2)
        # -x^2
        m2 = re.match(r'^(-?)([a-zA-Z])\^(\d+)$', inner)
        if m2:
            sign, var, power = m2.groups()
            return f"{m.group(1)}^{{{sign}{var}^{{{power}}}}}"
        return f"{m.group(1)}^{{{inner}}}"
    
    # sin(x)/x
    m = re.match(r'^(\w+)\((\w+)\)/(\w+)$', expr)
    if m:
        return f"\\{m.group(1)}({m.group(2)})/{m.group(3)}"
    
    return expr


def build_template(text: str) -> str | None:
    """Строит ожидаемый LaTeX из input."""
    text = text.strip()
    
    # =========================================================
    # СТЕПЕНИ: x^N
    # =========================================================
    m = re.match(r'^([a-zA-Z])\^(\d+)$', text)
    if m:
        return f"{m.group(1)}^{{{m.group(2)}}}"
    
    # =========================================================
    # ДРОБИ
    # =========================================================
    m = re.match(r'^(\d+)\s*/\s*(\d+)$', text)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    
    m = re.match(r'^([a-zA-Z])\s*/\s*([a-zA-Z])$', text)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    
    m = re.match(r'^(\w+)\s*/\s*(\w+)$', text)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    
    # =========================================================
    # КОРНИ
    # =========================================================
    m = re.match(r'^sqrt\((\w+)\)$', text)
    if m:
        return f"\\sqrt{{{m.group(1)}}}"
    
    m = re.match(r'^sqrt\((.+)\)$', text)
    if m:
        inner = m.group(1)
        m2 = re.match(r'^([a-zA-Z])\^(\d+)\s*\+\s*([a-zA-Z])\^(\d+)$', inner)
        if m2:
            a, na, b, nb = m2.groups()
            return f"\\sqrt{{{a}^{{{na}}} + {b}^{{{nb}}}}}"
        return f"\\sqrt{{{inner}}}"
    
    m = re.match(r'^cbrt\((\w+)\)$', text)
    if m:
        return f"\\sqrt[3]{{{m.group(1)}}}"
    
    # =========================================================
    # ГРЕЧЕСКИЕ
    # =========================================================
    m = re.match(r'^(\w+)\s*\+\s*(\w+)$', text)
    if m and m.group(1) in GREEK and m.group(2) in GREEK:
        return f"{GREEK[m.group(1)]} + {GREEK[m.group(2)]}"
    
    m = re.match(r'^(\w+)\s*-\s*(\w+)$', text)
    if m and m.group(1) in GREEK and m.group(2) in GREEK:
        return f"{GREEK[m.group(1)]} - {GREEK[m.group(2)]}"
    
    # =========================================================
    # СУММЫ КВАДРАТОВ
    # =========================================================
    m = re.match(r'^([a-zA-Z])\^(\d+)\s*\+\s*([a-zA-Z])\^(\d+)$', text)
    if m:
        a, na, b, nb = m.groups()
        return f"{a}^{{{na}}} + {b}^{{{nb}}}"
    
    m = re.match(r'^([a-zA-Z])\^(\d+)\s*\+\s*([a-zA-Z])\^(\d+)\s*=\s*([a-zA-Z])\^(\d+)$', text)
    if m:
        a, na, b, nb, c, nc = m.groups()
        return f"{a}^{{{na}}} + {b}^{{{nb}}} = {c}^{{{nc}}}"
    
    # =========================================================
    # ЛОГАРИФМЫ
    # =========================================================
    m = re.match(r'^(log|ln)\((\w+)\)$', text)
    if m:
        return f"\\{m.group(1)}({m.group(2)})"
    
    m = re.match(r'^(log|ln)\((.+)\)$', text)
    if m:
        return f"\\{m.group(1)}({m.group(2)})"
    
    m = re.match(r'^log_(\d+)\((\w+)\)$', text)
    if m:
        return f"\\log_{{{m.group(1)}}}({m.group(2)})"
    
    # =========================================================
    # ТРИГОНОМЕТРИЯ
    # =========================================================
    m = re.match(r'^(sin|cos|tan|arcsin|arccos|arctan)\((\w+)\)$', text)
    if m:
        return f"\\{m.group(1)}({m.group(2)})"
    
    m = re.match(r'^(sin|cos|tan)\((\d+)(\w+)\)$', text)
    if m:
        return f"\\{m.group(1)}({m.group(2)}{m.group(3)})"
    
    # =========================================================
    # ПРОИЗВОДНЫЕ
    # =========================================================
    m = re.match(r'^deriv\((.+)\)$', text)
    if m:
        inner = _convert_expr(m.group(1))
        return f"\\frac{{d}}{{dx}}({inner})"
    
    # =========================================================
    # ИНТЕГРАЛЫ
    # =========================================================
    m = re.match(r'^integral\s+(\S+)\s+(\S+)\s+(.+?)\s+dx$', text)
    if m:
        a, b, f = m.groups()
        a = GREEK_INF.get(a, a)
        b = GREEK_INF.get(b, b)
        f = _convert_expr(f)
        return f"\\int_{{{a}}}^{{{b}}} {f} \\, dx"
    
    m = re.match(r'^integral\s+(.+?)\s+dx$', text)
    if m:
        f = _convert_expr(m.group(1))
        return f"\\int {f} \\, dx"
    
    # =========================================================
    # СУММЫ
    # =========================================================
    m = re.match(r'^sum\s+(\w+)\s*=\s*(\S+)\s+(\S+)\s+(.+)$', text)
    if m:
        var, start, end, expr = m.groups()
        end = GREEK_INF.get(end, end)
        expr = _convert_expr(expr)
        return f"\\sum_{{{var}={start}}}^{{{end}}} {expr}"
    
    # =========================================================
    # ПРЕДЕЛЫ
    # =========================================================
    m = re.match(r'^lim\s+(\w+)\s*->\s*(\S+)\s+(.+)$', text)
    if m:
        var, target, expr = m.groups()
        target = GREEK_INF.get(target, target)
        # sin(x)/x
        m2 = re.match(r'^(\w+)\((\w+)\)/(\w+)$', expr)
        if m2:
            expr = f"\\{m2.group(1)}({m2.group(2)})/{m2.group(3)}"
        return f"\\lim_{{{var} \\to {target}}} {expr}"
    
    # =========================================================
    # УРАВНЕНИЯ: E = m c^2
    # =========================================================
    m = re.match(r'^([a-zA-Z])\s*=\s*([a-zA-Z])\s*([a-zA-Z])\^(\d+)$', text)
    if m:
        a, b, c, n = m.groups()
        return f"{a} = {b}{c}^{{{n}}}"
    
    m = re.match(r'^([a-zA-Z])\s*=\s*([a-zA-Z])\^(\d+)$', text)
    if m:
        a, b, n = m.groups()
        return f"{a} = {b}^{{{n}}}"
    
    # =========================================================
    # ПРОСТЫЕ ОПЕРАЦИИ
    # =========================================================
    m = re.match(r'^([a-zA-Z])\s*\+\s*([a-zA-Z])$', text)
    if m:
        return f"{m.group(1)} + {m.group(2)}"
    
    m = re.match(r'^([a-zA-Z])\s*-\s*([a-zA-Z])$', text)
    if m:
        return f"{m.group(1)} - {m.group(2)}"
    
    m = re.match(r'^([a-zA-Z])\s*\*\s*([a-zA-Z])$', text)
    if m:
        return f"{m.group(1)} \\cdot {m.group(2)}"
    
    return None


def postprocess(text: str, latex: str) -> str:
    """Template-based postprocessing."""
    template = build_template(text)
    if template is not None:
        return template
    return latex