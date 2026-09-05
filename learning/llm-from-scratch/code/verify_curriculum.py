"""Structural verification of the LLM curriculum before it moves into main."""
import os, re, sys

root = sys.argv[1]
mod = os.path.join(root, "modules")
build = os.path.join(mod, "build")

errors, warnings = [], []

REQUIRED = ["## Terms", "## Concepts", "## Where it's used", "## Labs",
            "## Failure modes", "## Interview", "## Checkpoint"]

def read(p):
    return open(p, encoding="utf-8").read()

# ---------- 1. every module has the fixed section structure ----------
ref = sorted(f for f in os.listdir(mod) if f.endswith(".md"))
bld = sorted(f for f in os.listdir(build) if f.endswith(".md") and f != "README.md")

for fn in ref:
    src = read(os.path.join(mod, fn))
    missing = [s for s in REQUIRED if s not in src]
    if missing and fn != "00-setup.md":
        warnings.append(f"modules/{fn}: missing sections {missing}")

for fn in bld:
    src = read(os.path.join(build, fn))
    missing = [s for s in REQUIRED if s not in src]
    if missing:
        errors.append(f"modules/build/{fn}: missing sections {missing}")

# ---------- 2. nav chain: every file has Next/Back, and Next targets exist ----------
def nav(dirpath, fn):
    src = read(os.path.join(dirpath, fn))
    tail = src[-800:]
    nxt = re.search(r"\*\*Next:\*\*\s*\[([^\]]+)\]\(([^)]+)\)", tail)
    bck = re.search(r"\*\*Back:\*\*\s*\[([^\]]+)\]\(([^)]+)\)", tail)
    return nxt, bck

for dirpath, files, label in ((mod, ref, "modules"), (build, bld, "modules/build")):
    for fn in files:
        if fn == "00-setup.md":
            continue
        nxt, bck = nav(dirpath, fn)
        if not nxt:
            warnings.append(f"{label}/{fn}: no **Next:** link")
        if not bck:
            warnings.append(f"{label}/{fn}: no **Back:** link")

# ---------- 3. build track forms an unbroken B0..B10 chain ----------
chain, cur, seen = [], "00-foundations.md", set()
while cur and cur not in seen:
    seen.add(cur); chain.append(cur)
    nxt, _ = nav(build, cur)
    if not nxt:
        break
    target = nxt.group(2).split("#")[0]
    if not target or not target.endswith(".md"):
        break
    cur = os.path.basename(target)
    if not os.path.exists(os.path.join(build, cur)):
        errors.append(f"build chain: {chain[-1]} -> missing {cur}")
        break
expected_chain = len(bld)
if len(chain) < expected_chain:
    errors.append(f"build chain reaches {len(chain)}/{expected_chain} lessons: {chain}")

# ---------- 4. every module appears in the top-level README ----------
readme = read(os.path.join(root, "SYLLABUS.md" if os.path.exists(os.path.join(root, "SYLLABUS.md")) else "README.md"))
for fn in ref:
    if fn == "00-setup.md":
        continue
    if f"modules/{fn}" not in readme:
        errors.append(f"modules/{fn} is not linked from README.md")
if "modules/build/README.md" not in readme:
    errors.append("Build Track is not linked from README.md")

# ---------- 5. build README lists every lesson ----------
broadme = read(os.path.join(build, "README.md"))
for fn in bld:
    if f"]({fn})" not in broadme:
        errors.append(f"modules/build/{fn} is not listed in the Build Track README")

# ---------- 6. relative links resolve ----------
pat = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
CODE_FALSE_POSITIVES = ("flat[rows]", "javascript:...")
for dirpath, _dirs, files in os.walk(root):
    if ".git" in dirpath or "node_modules" in dirpath:
        continue
    if not (dirpath.startswith(mod) or dirpath == root):
        continue
    for fn in files:
        if not fn.endswith(".md"):
            continue
        fp = os.path.join(dirpath, fn)
        for i, line in enumerate(read(fp).split("\n"), 1):
            for _text, target in pat.findall(line):
                if target.startswith(("http", "mailto:", "#")) or target in CODE_FALSE_POSITIVES:
                    continue
                path = target.split("#")[0]
                if not path:
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(dirpath, path))):
                    errors.append(f"{os.path.relpath(fp, root)}:{i} broken link -> {target}")

# ---------- 7. glossary sanity ----------
gl = read(os.path.join(root, "GLOSSARY.md"))
terms = re.findall(r"^\*\*(.+?)\*\*", gl, re.M)
dupes = {t for t in terms if terms.count(t) > 1}
if dupes:
    errors.append(f"GLOSSARY duplicate terms: {sorted(dupes)}")

# ---------- report ----------
print(f"reference modules : {len(ref)}")
print(f"build lessons     : {len(bld)}  chain: {' -> '.join(c[:2] for c in chain)}")
print(f"glossary terms    : {len(terms)}")
print(f"words             : ", end="")
wc = 0
for dirpath, _d, files in os.walk(root):
    if ".git" in dirpath or "node_modules" in dirpath or "/docs" in dirpath:
        continue
    for fn in files:
        if fn.endswith(".md"):
            wc += len(read(os.path.join(dirpath, fn)).split())
print(f"{wc:,}")
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print("  ✗", e)
else:
    print("no errors")
if warnings:
    print(f"\nwarnings ({len(warnings)}):")
    for w in warnings[:25]:
        print("  !", w)
sys.exit(1 if errors else 0)
