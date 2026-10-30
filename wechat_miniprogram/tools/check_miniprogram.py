# -*- coding: utf-8 -*-
"""Static gate for the mini program: compile-class checks without DevTools.

Validates: app.json page/tabBar paths exist on disk, every usingComponents
path resolves, WXML tags balance (ignoring self-closing), WXML event
handlers exist in the page JS, WXSS braces balance, all JSON parses.
Exit code 1 on any finding. Run: python tools/check_miniprogram.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MP = ROOT / "miniprogram"

SELF_CLOSE = re.compile(r"/>$")
TAG = re.compile(r"<(/?)([a-zA-Z][\w-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)(/?)>")
HANDLER = re.compile(r"\b(?:bind|catch)(?:[a-zA-Z]+)?=\"([A-Za-z_]\w*)\"")
JS_HANDLER = lambda name: re.compile(r"\b%s\s*(?::\s*function|\()" % re.escape(name))

VOID_TAGS = set()


def check_wxml(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    findings = []
    stack = []
    for m in TAG.finditer(text):
        closing, name, attrs, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if not closing and not selfclose and attrs.rstrip().endswith("/"):
            selfclose = "/"
        if closing:
            if not stack or stack[-1] != name:
                findings.append("%s: 标签闭合错误 </%s>（栈: %s）"
                                % (path.name, name, stack[-3:]))
                return findings
            stack.pop()
        elif not selfclose and name not in VOID_TAGS:
            stack.append(name)
    if stack:
        findings.append("%s: 未闭合标签 %s" % (path.name, stack))
    # every wxml handler must exist in the sibling js
    js = path.with_suffix(".js")
    js_text = js.read_text(encoding="utf-8") if js.exists() else ""
    for h in set(HANDLER.findall(text)):
        if h.startswith("tap"):
            continue
        if not JS_HANDLER(h).search(js_text):
            findings.append("%s: 事件处理器 %s 在 %s 中不存在"
                            % (path.name, h, js.name))
    return findings


def check_wxss(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    if text.count("{") != text.count("}"):
        return ["%s: 花括号不配对（{ %d 个，} %d 个）"
                % (path.name, text.count("{"), text.count("}"))]
    return []


def main():
    findings = []
    app_json = json.loads((MP / "app.json").read_text(encoding="utf-8"))
    pages = app_json.get("pages", [])
    for page in pages:
        for ext in ("js", "json", "wxml", "wxss"):
            if not (MP / ("%s.%s" % (page, ext))).is_file():
                findings.append("缺少页面文件 %s.%s" % (page, ext))
    tab_bar = app_json.get("tabBar")
    if tab_bar is not None:
        if not isinstance(tab_bar, dict):
            findings.append("app.json: tabBar 必须是对象（含 list 数组），当前是 %s"
                            % type(tab_bar).__name__)
        else:
            items = tab_bar.get("list")
            if not isinstance(items, list) or not 2 <= len(items) <= 5:
                findings.append("app.json: tabBar.list 必须是 2~5 项数组")
            else:
                for tab in items:
                    if tab.get("pagePath") not in pages:
                        findings.append("tabBar 页面不在 pages: %s"
                                        % tab.get("pagePath"))
                    if not tab.get("text"):
                        findings.append("tabBar 项缺少 text: %s"
                                        % tab.get("pagePath"))

    for jf in MP.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except ValueError as exc:
            findings.append("JSON 解析失败 %s: %s" % (jf.relative_to(ROOT), exc))
            continue
        for name, comp in (data.get("usingComponents") or {}).items():
            base = jf.parent
            target = (base / comp.strip("/")).resolve() if comp.startswith("/") \
                else (base / comp).resolve()
            # absolute paths are relative to miniprogram root
            if comp.startswith("/"):
                target = (MP / comp.lstrip("/")).resolve()
            if not target.with_suffix(".json").is_file():
                findings.append("组件路径无效 %s -> %s" % (jf.relative_to(ROOT), comp))

    for wf in MP.rglob("*.wxml"):
        findings += check_wxml(wf)
    for sf in MP.rglob("*.wxss"):
        findings += check_wxss(sf)

    if findings:
        print("MINIPROGRAM_CHECK_FAIL (%d)" % len(findings))
        for f in findings:
            print(" -", f)
        return 1
    print("MINIPROGRAM_CHECK_PASS: pages=%d wxml=%d wxss=%d json=%d"
          % (len(pages), len(list(MP.rglob("*.wxml"))),
             len(list(MP.rglob("*.wxss"))), len(list(MP.rglob("*.json")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
