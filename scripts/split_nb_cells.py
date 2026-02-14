#!/usr/bin/env python3
"""把 notebook 里混在 markdown 中的代码块拆成独立的 code / markdown cells。"""
import json
import re
from pathlib import Path


def split_markdown_into_cells(md_text: str) -> list[tuple[str, str]]:
    """按 ``` 代码块拆成 (type, content) 列表，交替 markdown / code。"""
    result: list[tuple[str, str]] = []
    rest = md_text
    while True:
        # 找第一个 ```（可能是 ```Python、```python 或 ```）
        m = re.search(r"```(?:[Pp]ython)?\s*\n", rest)
        if not m:
            if rest.strip():
                result.append(("markdown", rest.strip()))
            break
        # 代码块前的文字是 markdown
        before = rest[: m.start()].strip()
        if before:
            result.append(("markdown", before))
        # 代码从匹配结束到下一个 ```
        code_start = m.end()
        end_m = re.search(r"\n```\s*\n", rest[code_start:])
        if not end_m:
            # 没有结束 ```，剩余都当代码
            code = rest[code_start:].strip()
            if code:
                result.append(("code", code))
            break
        code = rest[code_start : code_start + end_m.start()].strip()
        if code:
            result.append(("code", code))
        rest = rest[code_start + end_m.end() :]
    return result


def source_from_text(text: str) -> list[str]:
    """把字符串转成 notebook source 数组（每行一个元素，末尾无多余换行）。"""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines:
        out.append(lines[-1] if lines[-1] else "\n")
    return out if out else [""]


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "notebooks" / "01_Single_Variable_Linear_Regression.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    new_cells: list[dict] = []

    for c in cells:
        ct = c.get("cell_type", "")
        src = c.get("source", [])
        text = "".join(s if isinstance(s, str) else "" for s in src)
        if ct == "markdown":
            has_fence = "```" in text
            print(f"  Cell: {ct}, len={len(text)}, has ``` = {has_fence}")
            segments = split_markdown_into_cells(text)
            if len(segments) > 1:
                print(f"  Markdown cell split into {len(segments)} segments.")
            for typ, content in segments:
                if not content.strip():
                    continue
                new_cells.append({
                    "cell_type": typ,
                    "metadata": dict(c.get("metadata", {})),
                    "source": source_from_text(content),
                })
        else:
            new_cells.append(dict(c))

    nb["cells"] = new_cells
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Split into {len(new_cells)} cells.")


if __name__ == "__main__":
    main()
