import re


def remove_duplicates(latex: str) -> str:
    """Убирает повторяющиеся подвыражения."""
    # \frac{...}{...}\frac{...}{...} → \frac{...}{...}
    latex = re.sub(r'(\\frac\{[^{}]+\}\{[^{}]+\})\1+', r'\1', latex)
    # \sqrt{...}\sqrt{...} → \sqrt{...}
    latex = re.sub(r'(\\sqrt\{[^{}]+\})\1+', r'\1', latex)
    # x^{N}x^{N} → x^{N}
    latex = re.sub(r'([a-zA-Z]\^\{[^{}]+\})\1+', r'\1', latex)
    # x_N x_N → x_N
    latex = re.sub(r'([a-zA-Z]_\{[^{}]+\})\1+', r'\1', latex)
    return latex


def balance_brackets(latex: str) -> str:
    """Балансирует фигурные скобки."""
    open_count = latex.count('{')
    close_count = latex.count('}')

    if close_count > open_count:
        excess = close_count - open_count
        for _ in range(excess):
            idx = latex.rfind('}')
            if idx != -1:
                latex = latex[:idx] + latex[idx+1:]
    elif open_count > close_count:
        latex += '}' * (open_count - close_count)

    return latex


def postprocess(latex: str) -> str:
    """Полная пост-обработка."""
    latex = remove_duplicates(latex)
    latex = balance_brackets(latex)
    latex = latex.strip()
    return latex


def safe_render(latex: str) -> tuple[str, bool]:
    """
    Возвращает (latex, valid).
    valid=True, если KaTeX может отрендерить.
    """
    try:
        import katex
        katex.renderToString(latex, throwOnError=True)
        return latex, True
    except Exception:
        return latex, False