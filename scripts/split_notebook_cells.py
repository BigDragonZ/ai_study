#!/usr/bin/env python3
"""
将 notebook 中「一个 markdown cell 内嵌多个 ``` 代码块」拆分为：
多个 markdown cell + 多个 code cell，结构清晰、Python 独立成 code cell。
"""
import json
import re
import uuid
from pathlib import Path


def split_markdown_by_code_fences(text: str) -> list[tuple[str, str]]:
    """
    按 ``` 代码块拆分，返回 [(cell_type, content), ...]，cell_type 为 'markdown' 或 'code'。
    """
    result: list[tuple[str, str]] = []
    # 匹配代码块开始：``` 或 ```Python 或 ```python，后接可选空白和换行
    pattern = re.compile(r"```(?:[Pp]ython)?\s*\n")
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            # 剩余全是 markdown
            rest = text[pos:].strip()
            if rest:
                result.append(("markdown", rest))
            break
        # 代码块前的 markdown
        before = text[pos : m.start()].strip()
        if before:
            result.append(("markdown", before))
        # 代码块内容：从匹配结束到下一个 \n```
        code_start = m.end()
        end_m = re.search(r"\n```\s*\n?", text[code_start:])
        if not end_m:
            code = text[code_start:].strip()
            if code:
                result.append(("code", code))
            break
        code = text[code_start : code_start + end_m.start()].strip()
        if code:
            result.append(("code", code))
        pos = code_start + end_m.end()
    return result


def to_source_lines(content: str, ensure_trailing_newline: bool = True) -> list[str]:
    """将字符串转为 notebook source 数组：除最后一行外每行末尾带 \\n。"""
    lines = content.split("\n")
    if not lines:
        return [""]
    out = [line + "\n" for line in lines[:-1]]
    last = lines[-1]
    if ensure_trailing_newline and last:
        out.append(last + "\n")
    else:
        out.append(last)
    return out


def main() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_Single_Variable_Linear_Regression.ipynb"
    )
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    if len(cells) != 1 or cells[0].get("cell_type") != "markdown":
        print("当前 notebook 不是「单个 markdown cell」结构，已跳过。")
        return

    full_text = "".join(
        s if isinstance(s, str) else "" for s in cells[0]["source"]
    )
    segments = split_markdown_by_code_fences(full_text)

    new_cells = []
    for i, (cell_type, content) in enumerate(segments):
        if not content.strip():
            continue
        cell = {
            "cell_type": cell_type,
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": to_source_lines(content),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        new_cells.append(cell)

    nb["cells"] = new_cells
    nb_path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"已拆分为 {len(new_cells)} 个 cell：")
    for i, c in enumerate(new_cells):
        print(f"  {i + 1}. {c['cell_type']} ({len(c['source'])} 行)")


if __name__ == "__main__":
    main()
