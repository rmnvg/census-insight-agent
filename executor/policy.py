import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

ALLOWED_IMPORTS = {
    "json",
    "math",
    "statistics",
    "decimal",
    "pathlib",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
}
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "system",
    "popen",
    "spawn",
    "fork",
    "connect",
    "request",
    "urlopen",
    "resolve",
    "absolute",
    "home",
    "cwd",
    "read_pickle",
    "to_pickle",
    "parent",
    "parents",
    "unlink",
    "rename",
    "replace",
    "rmdir",
    "mkdir",
    "chmod",
    "symlink_to",
    "hardlink_to",
    "touch",
    "glob",
    "rglob",
}
MAX_CODE_BYTES = 100_000
WRITE_METHODS = {"write_text", "write_bytes", "to_csv", "savefig", "save", "savetxt"}


@dataclass(frozen=True)
class PolicyResult:
    valid: bool
    errors: tuple[str, ...]


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, binding_counts: dict[str, int]) -> None:
        self.errors: list[str] = []
        self.known_paths: dict[str, PurePosixPath] = {}
        self.binding_counts = binding_counts

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._resolve_path(node.value)
        for target in node.targets:
            for name in self._assigned_names(target):
                self.known_paths.pop(name, None)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and resolved is not None
            and self.binding_counts.get(node.targets[0].id) == 1
        ):
            self.known_paths[node.targets[0].id] = resolved
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in self._assigned_names(node.target):
            self.known_paths.pop(name, None)
        resolved = self._resolve_path(node.value)
        if (
            isinstance(node.target, ast.Name)
            and resolved is not None
            and self.binding_counts.get(node.target.id) == 1
        ):
            self.known_paths[node.target.id] = resolved
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in self._assigned_names(node.target):
            self.known_paths.pop(name, None)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        for name in self._assigned_names(node.target):
            self.known_paths.pop(name, None)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for name in self._assigned_names(node.target):
            self.known_paths.pop(name, None)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        for name in self._assigned_names(node.target):
            self.known_paths.pop(name, None)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._invalidate_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._invalidate_arguments(node.args)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                self.errors.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module.split(".")[0] not in ALLOWED_IMPORTS:
            self.errors.append(f"Forbidden import: {module or 'relative'}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden call: {node.func.id}")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._validate_open_call(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTES:
            safe_output_mkdir = node.func.attr == "mkdir" and self._resolve_path(
                node.func.value
            ) == PurePosixPath("output")
            if not safe_output_mkdir:
                self.errors.append(f"Forbidden attribute call: {node.func.attr}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in WRITE_METHODS:
            path_node = (
                node.func.value
                if node.func.attr in {"write_text", "write_bytes"}
                else node.args[0]
                if node.args
                else None
            )
            if not self._is_output_path(path_node):
                self.errors.append(f"Write path must be a literal output path: {node.func.attr}")
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            self._validate_path_argument(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr.endswith("__"):
            self.errors.append(f"Dunder access is forbidden: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.errors.append(f"Dunder name is forbidden: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            path = PurePosixPath(node.value)
            if path.is_absolute() or ".." in path.parts:
                self.errors.append(f"Unsafe path literal: {node.value}")

    def _validate_path_argument(self, node: ast.Call) -> None:
        if not node.args:
            self.errors.append("Path arguments must be literal or statically known relative paths")
            return
        path = self._resolve_path(node.args[0])
        if path is None:
            self.errors.append("Path arguments must be literal or statically known relative paths")
            return
        if path.is_absolute() or ".." in path.parts:
            self.errors.append(f"Unsafe path: {path}")
        elif path != PurePosixPath("input.json") and path.parts[:1] != ("output",):
            self.errors.append(f"Path outside job inputs/outputs: {path}")

    def _validate_open_call(self, node: ast.Call) -> None:
        path_node = node.args[0] if node.args else self._keyword_value(node, "file")
        mode_node = node.args[1] if len(node.args) > 1 else self._keyword_value(node, "mode")
        path = self._resolve_path(path_node)
        mode = "r" if mode_node is None else self._constant_string(mode_node)
        if path is None or mode is None:
            self.errors.append("open requires a statically known job path and constant mode")
            return
        if mode in {"r", "rt", "rb"} and path == PurePosixPath("input.json"):
            return
        if mode in {"w", "wt", "wb"} and self._is_safe_output_path(path):
            return
        self.errors.append("open may only read input.json or write a statically known output path")

    def _is_output_path(self, node: ast.AST | None) -> bool:
        path = self._resolve_path(node)
        return path is not None and self._is_safe_output_path(path)

    @staticmethod
    def _is_safe_output_path(path: PurePosixPath) -> bool:
        return (
            not path.is_absolute()
            and ".." not in path.parts
            and len(path.parts) > 1
            and path.parts[:1] == ("output",)
        )

    def _resolve_path(self, node: ast.AST | None) -> PurePosixPath | None:
        if isinstance(node, ast.Name):
            return self.known_paths.get(node.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
        ):
            return self._resolve_path(node.args[0] if node.args else None)
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            path = self._resolve_path(node.value)
            return path.parent if path is not None else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # `Path("output") / "table.csv"` is the standard pathlib join idiom and is exactly
            # as statically analyzable as a single string literal: both operands still have to
            # resolve through this same strict recursion, and the combined result still passes
            # through the unchanged _is_safe_output_path/_validate_path_argument checks below.
            left = self._resolve_path(node.left)
            right = self._resolve_path(node.right)
            return left / right if left is not None and right is not None else None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return PurePosixPath(node.value)
        return None

    @staticmethod
    def _assigned_names(node: ast.AST) -> set[str]:
        return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}

    def _invalidate_arguments(self, arguments: ast.arguments) -> None:
        for argument in [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]:
            self.known_paths.pop(argument.arg, None)
        if arguments.vararg is not None:
            self.known_paths.pop(arguments.vararg.arg, None)
        if arguments.kwarg is not None:
            self.known_paths.pop(arguments.kwarg.arg, None)

    @staticmethod
    def _keyword_value(node: ast.Call, name: str) -> ast.AST | None:
        return next((item.value for item in node.keywords if item.arg == name), None)

    @staticmethod
    def _constant_string(node: ast.AST) -> str | None:
        return (
            node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
        )


def validate_code(code: str) -> PolicyResult:
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return PolicyResult(False, ("Source exceeds maximum size",))
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return PolicyResult(False, (f"Syntax error: {error.msg}",))
    visitor = _PolicyVisitor(_binding_counts(tree))
    visitor.visit(tree)
    return PolicyResult(not visitor.errors, tuple(dict.fromkeys(visitor.errors)))


def _binding_counts(tree: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add(node.id)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name)
    return counts
