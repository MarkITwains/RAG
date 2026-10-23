# -*- coding: utf-8 -*-
"""
整理 data/clear_docs 与 data/nonFixed_docs -> data/pcb_docs_clean
只读原目录，不删除、不修改任何原始文件。
"""
import os, re, json, collections, shutil

ROOT = r"D:/myProject/rag_v1-main/rag_v1/data"
OUT = os.path.join(ROOT, "pcb_docs_clean")
SRC_DIRS = ["clear_docs", "nonFixed_docs"]

NAME_FIX = [
    ("覆铜结", "覆铜箔"), ("聚酷亚肢", "聚酰亚胺"), ("聚醋", "聚酯"),
    ("Candence", "Cadence"), ("（PDF）", ""), ("(PDF)", ""),
]

def fix_name(s):
    for a, b in NAME_FIX:
        s = s.replace(a, b)
    return s

def load(p):
    return open(p, encoding="utf-8", errors="replace").read()

def cjk_count(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t))

SEC = re.compile(r"^\s*(\d{1,2}(?:[.．·，,]\s?\d{1,2}){0,2})\s*[\u4e00-\u9fff]")

def body_sections(t):
    s = set()
    for line in t.splitlines():
        m = SEC.match(line)
        if m:
            s.add(re.sub(r"[．·，,]\s?", ".", m.group(1)))
    return s

def norm_key(name):
    s = fix_name(name).upper().replace(" ", "")
    m = re.search(r"GB/?T(\d{1,5}(?:\.\d{1,2})?)(?:-(\d{4}))?", s)
    if m:
        return "gbt%s%s" % (m.group(1), ("-" + m.group(2)) if m.group(2) else "")
    m = re.search(r"IPC[-_]?(\d{3,4}[A-Z]*)", s)
    if m:
        return "ipc" + m.group(1)
    return None

BOOK_KEYS = ("ALTIUM", "CADENCE", "ALLEGRO", "教程", "宝典", "攻略", "设计与仿真", "信号完整性")

def is_book(name):
    u = fix_name(name).upper()
    return any(k.upper() in u for k in BOOK_KEYS)

def book_key(name):
    s = fix_name(os.path.splitext(name)[0])
    s = re.sub(r"[-_]\s*\d{1,2}$", "", s)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    return s.strip()

def clean_title(name):
    s = fix_name(os.path.splitext(name)[0])
    s = re.sub(r"^\s*修改版\s*", "", s)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = s.replace("GBT4588.4-20171", "GBT4588.4-2017")
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s*标准\s*$", "", s)
    s = s.strip(" _")
    # 去掉中文后紧跟的多余尾号（如 通用规范1 -> 通用规范）
    s = re.sub(r"(?<=[\u4e00-\u9fff])\d{1,2}$", "", s).strip(" _")
    m = re.match(r"^GB\s*/?\s*T\s*(\d{1,5}(?:\.\d{1,2})?)\s*[-—]\s*(\d{4})\s*(.*)$", s)
    if m:
        num, year, rest = m.group(1), m.group(2), m.group(3).strip(" _")
        s = "GBT%s-%s%s" % (num, year, ("_" + rest) if rest else "")
    m = re.match(r"^IPC[\s_-]*(\d{3,4}[A-Za-z]*)\s*(.*)$", s, re.I)
    if m:
        s = "IPC-%s%s" % (m.group(1).upper(), ("_" + m.group(2).strip(" _")) if m.group(2).strip(" _") else "")
    return re.sub(r"[\\/:*?\"<>|]", "-", s).strip()

# ---------- 收集 ----------
records = []
for d in SRC_DIRS:
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    for name in sorted(os.listdir(dp)):
        p = os.path.join(dp, name)
        if not os.path.isfile(p) or not name.lower().endswith(".txt"):
            continue
        t = load(p)
        bk = book_key(name) if is_book(name) else None
        records.append({
            "src_dir": d, "src_name": name, "src_path": p,
            "chars": len(t), "cjk": cjk_count(t), "secs": body_sections(t),
            "key": ("BOOK::" + bk.lower()) if bk else (norm_key(name) or ("FILE::" + name)),
            "text": t,
        })

groups = collections.defaultdict(list)
for r in records:
    groups[r["key"]].append(r)

def pick_best(cands):
    best = max(cands, key=lambda r: (len(r["secs"]), r["cjk"]))
    clear = [r for r in cands if r["src_dir"] == "clear_docs"]
    if clear:
        c = max(clear, key=lambda r: (len(r["secs"]), r["cjk"]))
        if c["cjk"] >= 0.75 * best["cjk"]:
            return c, "clear_docs(清洗版)", []
        return best, "nonFixed_docs(更完整,需复核)", ["采用非清洗版：clear 版正文疑似不全，建议人工复核"]
    return best, "nonFixed_docs(唯一来源)", ["clear_docs 中无对应文件；含封面/目次"]

# ---------- 输出 ----------
if os.path.isdir(OUT) and os.path.basename(OUT) == "pcb_docs_clean":
    shutil.rmtree(OUT)
for sub in ("01_国家标准", "02_教材与专著", "03_国际标准", "04_其他"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

manifest, used = [], set()

def unique(sub, base):
    name, i = base + ".txt", 1
    while os.path.join(sub, name) in used:
        i += 1
        name = "%s_%d.txt" % (base, i)
    used.add(os.path.join(sub, name))
    return name

for key, cands in sorted(groups.items()):
    if key.startswith("BOOK::"):
        frags = sorted(cands, key=lambda r: (
            int(re.search(r"[-_]\s*(\d{1,2})$", os.path.splitext(r["src_name"])[0]).group(1))
            if re.search(r"[-_]\s*(\d{1,2})$", os.path.splitext(r["src_name"])[0]) else 0))
        body = "\n\n".join(r["text"].strip() for r in frags)
        base = re.sub(r"[-_]\s*\d{1,2}$", "", clean_title(frags[0]["src_name"]))
        sub = "02_教材与专著"
        name = unique(sub, base)
        open(os.path.join(OUT, sub, name), "w", encoding="utf-8", newline="\n").write(body)
        manifest.append({
            "out": sub + "/" + name, "chars": len(body), "cjk": cjk_count(body),
            "sections": len(body_sections(body)), "source": "merged:%d" % len(frags),
            "chosen_by": "分卷合并" if len(frags) > 1 else "单文件",
            "candidates": [r["src_dir"] + "/" + r["src_name"] for r in frags],
            "flags": (["由 %d 个分卷合并，按卷号升序，段落连续" % len(frags)] if len(frags) > 1 else []),
            "group": key,
        })
        continue

    best, why, flags = pick_best(cands)
    title = clean_title(best["src_name"])
    if title.startswith("IPC"):
        sub = "03_国际标准"
    elif title.startswith("GBT"):
        sub = "01_国家标准"
    else:
        sub = "04_其他"
    name = unique(sub, title)
    open(os.path.join(OUT, sub, name), "w", encoding="utf-8", newline="\n").write(best["text"])
    manifest.append({
        "out": sub + "/" + name, "chars": best["chars"], "cjk": best["cjk"],
        "sections": len(best["secs"]), "source": best["src_dir"] + "/" + best["src_name"],
        "chosen_by": why,
        "candidates": [c["src_dir"] + "/" + c["src_name"] + "(%d)" % c["chars"] for c in cands],
        "flags": flags, "group": key,
    })

manifest.sort(key=lambda m: m["out"])
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# 清理空目录
for sub in ("01_国家标准", "02_教材与专著", "03_国际标准", "04_其他"):
    p = os.path.join(OUT, sub)
    if os.path.isdir(p) and not os.listdir(p):
        os.rmdir(p)

print("files:", len(manifest), " chars:", sum(m["chars"] for m in manifest),
      " cjk:", sum(m["cjk"] for m in manifest))
print()
for m in manifest:
    print("%-70s %7d %-26s %s" % (m["out"][:70], m["chars"], m["chosen_by"], ";".join(m["flags"])))
