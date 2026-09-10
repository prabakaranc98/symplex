"""The importable scientific library shipped into the sandbox as one file.

Every hosted run currently starts from nothing. The agent re-derives an integrator, a
sensitivity sweep, a calibration score -- badly, differently each time, and with no way
for a reader to tell this run's Runge-Kutta from last run's. That is not authorship; it
is repetition with fresh bugs. This module collects the pure-numeric Symplex modules
into a single `symplex_sci.py` that agent-authored sandbox code can simply import.

Three rules govern what goes in:

* **Discovery is by filesystem path, never by import.** Modules elsewhere in this
  repository are being written concurrently. A candidate that does not exist yet, does
  not parse, or reaches for a package the sandbox does not have is skipped and reported.
  The bundle never hard-fails because a module is missing.
* **The bundle is content-addressed.** `bundle_digest()` binds the exact library version
  into run identity, the same discipline `symplex/connectors/compute.py` applies to
  every uploaded input.
* **The library is an offer, not an obligation.** The generated header says so
  explicitly, and so does `api_reference()`. An agent that believes the bundled routine
  is wrong for its problem should write its own and say why -- that judgement is the
  agent's, and taking it away would defeat the point of the agent authoring the model.

`bundle_digest` returns a bare digest string, matching `symplex.core.contracts.digest`;
every other public function here returns a scoped dict.
"""

import ast
import hashlib
import pprint
from pathlib import Path

from symplex.connectors.compute import MAX_INPUT_BYTES, MAX_REVISION_FILES
from symplex.core.contracts import Invalid, digest

VERSION = "symplex-sci-bundle-v1"
BUNDLE_FILENAME = "symplex_sci.py"
MAX_MODULES = 24
MAX_MODULE_BYTES = 400_000
MAX_BUNDLE_BYTES = min(700_000, MAX_INPUT_BYTES)
MAX_API_CHARS = 6000
MAX_SCANNED_FILES = 200

SCOPE = (
    "A code library assembled from repository sources for import inside the sandbox. "
    "Importing it establishes nothing scientific: the routines are host-written code, "
    "not validated methods, and an agent remains free to implement its own instead."
)

# Third-party roots assumed present in the hosted Python sandbox. Anything outside this
# set makes a module unbundleable, because a single missing import would break the whole
# single-file bundle for every agent that imports it.
SANDBOX_PACKAGES = frozenset({"numpy", "scipy"})

# Standard-library modules a pure numeric routine legitimately needs. Host-facing
# modules (os, pathlib, subprocess, socket, importlib, sqlite3, tempfile, shutil) are
# deliberately absent: a module that reaches for them is host infrastructure, not a
# sandbox library, and would try to touch a filesystem that is not there.
PURE_STDLIB = frozenset(
    {
        "abc", "array", "ast", "base64", "binascii", "bisect", "cmath", "collections",
        "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
        "fractions", "functools", "hashlib", "heapq", "inspect", "io", "itertools",
        "json", "math", "numbers", "operator", "random", "re", "statistics", "string",
        "__future__",
        "struct", "sys", "textwrap", "time", "types", "typing", "unicodedata", "uuid",
        "warnings", "zlib",
    }
)

# Ordered candidates. Explicit files first, then whole packages scanned by path. Order
# is a preference, not a dependency graph; dependencies are resolved from the sources.
CANDIDATE_PATHS = ("symplex/core/contracts.py",)
CANDIDATE_DIRECTORIES = (
    "symplex/semantics",
    "symplex/modeling",
    "symplex/inference",
    "symplex/hybrid",
    "symplex/verticals",
)
# symplex/evaluation is deliberately absent. Host-owned recomputation is the one thing
# the maker never receives; shipping it into the sandbox would blur the only boundary
# that makes host verification worth anything.
SKIP_BASENAMES = frozenset({"__init__.py", "__main__.py", "conftest.py"})

HEADER = '''"""symplex_sci -- Symplex scientific routines, bundled for this sandbox run.

YOU MAY IMPORT THIS. YOU ARE NOT REQUIRED TO.

    from symplex_sci import SYMPLEX_SCI_MANIFEST   # what is in here
    from symplex_sci import <function>             # any name listed in the manifest

This file exists so you do not have to re-derive an integrator, an estimator or a
calibration score from scratch on every run, and so a reader can see which routine you
actually used. It is ordinary host-written code. It is not validated science, it carries
no authority, and importing a function from it does not make a result correct.

If a bundled routine is wrong for your problem -- wrong assumptions, wrong regime, wrong
numerics -- write your own and say in your results why the bundled one did not fit.
That judgement is yours. A worse model that imported from here is not better than a
better model that did not.

Provenance: assembled from repository sources by symplex/authoring/bundle.py.
Bundle version: {version}
Bundle digest: {bundle_digest}
Modules: {module_names}
Omitted: {omitted_names}
"""

SYMPLEX_SCI_MANIFEST = {manifest}
'''


def _root(root):
    base = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    if not base.is_dir():
        raise Invalid("Bundle root is not a directory")
    return base


def _candidates(base, modules):
    """Discover candidate sources by path. Absent paths are simply absent."""
    if modules is not None:
        if not isinstance(modules, (list, tuple)) or len(modules) > MAX_SCANNED_FILES:
            raise Invalid("Supply candidate module paths as a bounded list")
        requested = []
        for item in modules:
            if not isinstance(item, str) or not item.strip():
                raise Invalid("Candidate module paths must be nonempty text")
            path = Path(item)
            requested.append(path if path.is_absolute() else base / path)
        return requested
    found = [base / p for p in CANDIDATE_PATHS]
    for directory in CANDIDATE_DIRECTORIES:
        folder = base / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts or path.name in SKIP_BASENAMES:
                continue
            found.append(path)
            if len(found) >= MAX_SCANNED_FILES:
                return found
    return found


def _module_name(base, path):
    try:
        relative = path.resolve().relative_to(base)
    except ValueError:
        return path.stem
    return ".".join(relative.with_suffix("").parts)


def _imports(tree, module_name):
    """Collect (root, resolved_internal_name, node) for every import in the module."""
    package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    external, internal, nodes = set(), set(), []
    future = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "symplex":
                    internal.add(alias.name)
                    nodes.append(node)
                else:
                    external.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                target = ".".join(parts[: len(parts) - node.level + 1] + ([node.module] if node.module else []))
                internal.add(target)
                nodes.append(node)
                continue
            root = (node.module or "").split(".")[0]
            if root == "symplex":
                internal.add(node.module)
                nodes.append(node)
            elif root:
                external.add(root)
    return external, internal, nodes, future


def _signature(node):
    args = node.args
    parts = []

    def render(arg, default=None):
        text = arg.arg
        if arg.annotation is not None:
            try:
                text += ": " + ast.unparse(arg.annotation)
            except Exception:
                pass
        if default is not None:
            try:
                text += "=" + ast.unparse(default)
            except Exception:
                text += "=..."
        return text

    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, arg in enumerate(positional):
        parts.append(render(arg, defaults[index]))
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(render(arg, default))
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)
    return node.name + "(" + ", ".join(parts) + ")"


def _api(tree):
    entries = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            summary = (ast.get_docstring(node) or "").strip().splitlines()
            entries.append(
                {
                    "symbol": _signature(node),
                    "kind": "function",
                    "summary": (summary[0] if summary else "")[:200],
                }
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            summary = (ast.get_docstring(node) or "").strip().splitlines()
            methods = [
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not m.name.startswith("_")
            ][:12]
            entries.append(
                {
                    "symbol": node.name,
                    "kind": "class",
                    "methods": methods,
                    "summary": (summary[0] if summary else "")[:200],
                }
            )
    return entries[:80]


def _neutralize(source, nodes):
    """Replace intra-repository import lines; every bundled name shares one namespace."""
    lines = source.splitlines()
    for node in nodes:
        start = getattr(node, "lineno", 0) - 1
        end = getattr(node, "end_lineno", node.lineno) - 1
        if start < 0 or end >= len(lines):
            continue
        indent = " " * getattr(node, "col_offset", 0)
        try:
            described = ast.unparse(node).replace("\n", " ")[:160]
        except Exception:
            described = "intra-repository import"
        lines[start] = indent + "pass  # bundled in this file: " + described
        for index in range(start + 1, end + 1):
            lines[index] = indent + "# " + lines[index].strip()[:200]
    return "\n".join(lines)


def _read(path):
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "unreadable"
    if len(raw) > MAX_MODULE_BYTES:
        return None, "module exceeds the per-module byte ceiling"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "module is not valid UTF-8"


def _scan(base, paths, available):
    """Parse every candidate and decide, per module, whether it can be bundled."""
    scanned, omitted, seen = {}, [], set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        name = _module_name(base, path)
        if not path.is_file():
            omitted.append({"module": name, "path": key, "reason": "module does not exist"})
            continue
        source, problem = _read(path)
        if source is None:
            omitted.append({"module": name, "path": key, "reason": problem})
            continue
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError as exc:
            omitted.append(
                {"module": name, "path": key, "reason": "does not parse: " + str(exc)[:200]}
            )
            continue
        external, internal, nodes, future = _imports(tree, name)
        if future:
            omitted.append(
                {
                    "module": name,
                    "path": key,
                    "reason": "uses a __future__ statement, which cannot be relocated into a concatenated bundle",
                }
            )
            continue
        unavailable = sorted(external - available)
        if unavailable:
            omitted.append(
                {
                    "module": name,
                    "path": key,
                    "reason": "imports packages unavailable in the sandbox: "
                    + ", ".join(unavailable[:8]),
                }
            )
            continue
        scanned[name] = {
            "module": name,
            "path": key,
            "source": source,
            "tree": tree,
            "nodes": nodes,
            "internal": sorted(internal),
            "requires": sorted(external),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "bytes": len(source.encode()),
        }
    return scanned, omitted


def _resolve(scanned, omitted):
    """Drop modules whose intra-repository dependencies are not themselves bundleable."""
    admitted = dict(scanned)
    changed = True
    while changed:
        changed = False
        for name in sorted(admitted):
            missing = [d for d in admitted[name]["internal"] if d not in admitted]
            if missing:
                omitted.append(
                    {
                        "module": name,
                        "path": admitted[name]["path"],
                        "reason": "depends on modules that cannot be bundled: "
                        + ", ".join(sorted(missing)[:6]),
                    }
                )
                del admitted[name]
                changed = True
                break
    return admitted


def _order(admitted, omitted):
    """Dependency-first ordering, deterministic in the presence of cycles."""
    ordered, visiting, done = [], set(), set()
    cycles = []

    def visit(name):
        if name in done:
            return
        if name in visiting:
            cycles.append(name)
            return
        visiting.add(name)
        for dependency in sorted(admitted[name]["internal"]):
            if dependency in admitted:
                visit(dependency)
        visiting.discard(name)
        done.add(name)
        ordered.append(name)

    for name in sorted(admitted):
        visit(name)
    if cycles:
        omitted.append(
            {
                "module": ", ".join(sorted(set(cycles))[:6]),
                "path": "",
                "reason": "import cycle; bundled in name order, which may not resolve at import time",
            }
        )
    return ordered


def build_bundle(modules=None, *, root=None, available_packages=None, max_bytes=None):
    """Assemble the pure-numeric Symplex modules into one uploadable `symplex_sci.py`.

    Candidates are discovered by filesystem path. A candidate is bundled only if it
    parses, stays inside the per-module byte ceiling, imports nothing outside the
    sandbox package set, and depends only on other bundled modules. Anything else is
    reported in `omitted` with a reason and the build continues -- a module that another
    part of this repository has not written yet must never break the loop that ships the
    library.

    Returns a dict carrying the assembled `source`, its `sha256`, the `manifest`, and the
    `bundle_digest` that binds this library version into run provenance. The bundle is
    code, not evidence; nothing here validates a routine or a result.
    """
    base = _root(root)
    ceiling = MAX_BUNDLE_BYTES if max_bytes is None else int(max_bytes)
    if not 0 < ceiling <= MAX_INPUT_BYTES:
        raise Invalid("Bundle byte ceiling must be positive and within the compute input envelope")
    available = frozenset(available_packages) if available_packages is not None else SANDBOX_PACKAGES
    available = available | PURE_STDLIB

    scanned, omitted = _scan(base, _candidates(base, modules), available)
    admitted = _resolve(scanned, omitted)
    ordered = _order(admitted, omitted)

    selected, dropped, size = [], set(), 0
    for name in ordered:
        entry = admitted[name]
        blocked = [d for d in entry["internal"] if d in dropped]
        if blocked:
            dropped.add(name)
            omitted.append(
                {
                    "module": name,
                    "path": entry["path"],
                    "reason": "dropped with its dependency to stay inside the bundle envelope",
                }
            )
            continue
        if len(selected) >= MAX_MODULES or size + entry["bytes"] > ceiling:
            dropped.add(name)
            omitted.append(
                {
                    "module": name,
                    "path": entry["path"],
                    "reason": "bundle module or byte envelope exceeded",
                }
            )
            continue
        selected.append(name)
        size += entry["bytes"]

    defined, shadowed = {}, []
    module_records = []
    for name in selected:
        entry = admitted[name]
        api = _api(entry["tree"])
        for item in api:
            symbol = item["symbol"].split("(")[0]
            if symbol in defined and defined[symbol] != name:
                shadowed.append(
                    {"symbol": symbol, "defined_in": defined[symbol], "shadowed_by": name}
                )
            defined[symbol] = name
        module_records.append(
            {
                "module": name,
                "path": str(Path(entry["path"]).relative_to(base))
                if Path(entry["path"]).is_relative_to(base)
                else entry["path"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "requires": entry["requires"],
                "api": api,
            }
        )

    manifest = {
        "version": VERSION,
        "modules": [
            {k: v for k, v in record.items() if k != "api"} for record in module_records
        ],
        "api": {record["module"]: record["api"] for record in module_records},
        "shadowed_names": shadowed[:32],
        "omitted": [
            {"module": o["module"], "reason": o["reason"]} for o in omitted
        ][:64],
        "authority": "host_written_library",
        "scope": SCOPE,
    }
    bundle_id = digest({"version": VERSION, "modules": manifest["modules"]})

    body = [
        HEADER.format(
            version=VERSION,
            bundle_digest=bundle_id,
            module_names=", ".join(selected) or "none",
            omitted_names=str(len(omitted)) + " candidate(s) skipped",
            manifest=pprint.pformat(manifest, width=100, sort_dicts=True),
        )
    ]
    for name in selected:
        entry = admitted[name]
        body.append(
            "\n\n# "
            + "=" * 76
            + f"\n# begin {name}  sha256={entry['sha256'][:16]}\n# "
            + "=" * 76
            + "\n"
            + _neutralize(entry["source"], entry["nodes"])
            + f"\n# end {name}\n"
        )
    source = "".join(body)

    status = "built" if selected else "empty"
    try:
        compile(source, BUNDLE_FILENAME, "exec")
    except SyntaxError as exc:
        return _unavailable(
            manifest, bundle_id, omitted, "assembled bundle does not compile: " + str(exc)[:300]
        )
    raw = source.encode()
    if len(raw) > ceiling:
        return _unavailable(manifest, bundle_id, omitted, "assembled bundle exceeds its byte ceiling")

    return {
        "status": status,
        "filename": BUNDLE_FILENAME,
        "source": source,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bundle_digest": bundle_id,
        "manifest": manifest,
        "modules": module_records,
        "module_names": selected,
        "omitted": omitted[:64],
        "omitted_count": len(omitted),
        "shadowed_names": shadowed[:32],
        "file_count": 1,
        "max_files": MAX_REVISION_FILES,
        "max_bytes": ceiling,
        "root": str(base),
        "agent_may_ignore": (
            "The agent may import from this bundle or implement its own routine. Importing "
            "from it confers no correctness, and declining it is not a defect."
        ),
        "independently_validated": False,
        "scope": SCOPE,
    }


def _unavailable(manifest, bundle_id, omitted, reason):
    return {
        "status": "unavailable",
        "filename": BUNDLE_FILENAME,
        "source": "",
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "bundle_digest": bundle_id,
        "manifest": manifest,
        "modules": [],
        "module_names": [],
        "omitted": (omitted + [{"module": "", "path": "", "reason": reason}])[:64],
        "omitted_count": len(omitted) + 1,
        "shadowed_names": [],
        "file_count": 0,
        "max_files": MAX_REVISION_FILES,
        "max_bytes": MAX_BUNDLE_BYTES,
        "root": "",
        "reason": reason,
        "agent_may_ignore": "No library is available for this run; write the routines you need.",
        "independently_validated": False,
        "scope": SCOPE,
    }


def bundle_digest(modules=None, *, root=None, available_packages=None, max_bytes=None):
    """Content digest of the bundle, so the library version binds into run identity.

    Returns a bare digest string, matching `symplex.core.contracts.digest`, because it is
    an identity accessor rather than a result. It is stable for identical sources and
    changes whenever any bundled module's bytes change, exactly like the input digests
    `symplex/connectors/compute.py` records for every uploaded file.
    """
    return build_bundle(
        modules, root=root, available_packages=available_packages, max_bytes=max_bytes
    )["bundle_digest"]


def api_reference(bundle=None, *, max_chars=MAX_API_CHARS, **kwargs):
    """A compact, token-bounded signature listing for the agent's prompt context.

    It tells the agent what it MAY import, and says in the same breath that it remains
    free to write its own implementation. Truncation is reported rather than hidden: a
    silently shortened API listing would make the agent think a routine does not exist.
    """
    built = bundle if bundle is not None else build_bundle(**kwargs)
    if not isinstance(built, dict) or "manifest" not in built:
        raise Invalid("Supply a bundle produced by build_bundle")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or not 200 <= max_chars <= 40000:
        raise Invalid("api_reference character bound must be between 200 and 40000")

    lines = [
        f"# symplex_sci  ({built.get('status', 'unknown')}, digest {built['bundle_digest'][:16]})",
        "# Import any of these inside the sandbox:  from symplex_sci import <name>",
        "# You are free to ignore all of it and write your own implementation. Say so if you do.",
    ]
    truncated = False
    for record in built.get("modules", []):
        header = f"\n## {record['module']}  [requires: {', '.join(record['requires']) or 'stdlib only'}]"
        block = [header]
        for item in record.get("api", []):
            prefix = "class " if item["kind"] == "class" else "def "
            entry = "  " + prefix + item["symbol"]
            if item.get("methods"):
                entry += "  methods: " + ", ".join(item["methods"])
            if item.get("summary"):
                entry += "\n      " + item["summary"]
            block.append(entry)
        candidate = "\n".join(lines + block)
        if len(candidate) > max_chars:
            truncated = True
            break
        lines.extend(block)
    if truncated:
        lines.append("\n# (listing truncated; further modules exist in SYMPLEX_SCI_MANIFEST)")
    text = "\n".join(lines)[:max_chars]
    return {
        "status": built.get("status", "unknown"),
        "text": text,
        "characters": len(text),
        "truncated": truncated,
        "module_names": built.get("module_names", []),
        "omitted_count": built.get("omitted_count", 0),
        "bundle_digest": built["bundle_digest"],
        "filename": BUNDLE_FILENAME,
        "agent_may_ignore": built.get("agent_may_ignore", ""),
        "independently_validated": False,
        "scope": SCOPE,
    }
