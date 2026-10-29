"""Regression checks for independent entry points and documented execution."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_launcher", ROOT / "run_experiments.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PortabilityTests(unittest.TestCase):
    def test_all_launcher_flags_exist_in_target_parsers(self):
        for stage in ["core", "lr", "head", "prediction", "solvers"]:
            for command in launcher.commands(stage, 1):
                source = Path(command[1])
                self.assertTrue(source.is_file(), source)
                tree = ast.parse(source.read_text(encoding="utf-8"))
                accepted = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                        accepted.update(arg.value for arg in node.args
                                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                        and arg.value.startswith("--"))
                supplied = {a for a in command[2:] if a.startswith("--")}
                self.assertFalse(supplied - accepted, (source, supplied - accepted))

    def test_dry_run_works_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="dp_sgd_foreign_cwd_") as directory:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "run_experiments.py"),
                                     "all", "--dry-run"], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]
        self.assertTrue(commands)
        tex_jobs = [job for job in commands if job[0] == "pdflatex"]
        self.assertEqual(len(tex_jobs), 10)
        for job in tex_jobs:
            self.assertNotIn(str(ROOT), job[-1])
            self.assertTrue(job[-1].isascii())
        for job in commands:
            if job[0] == sys.executable:
                self.assertTrue(Path(job[1]).is_relative_to(ROOT))
                self.assertTrue(Path(job[1]).is_file())

    def test_numerical_only_command_has_no_tex_jobs(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run_experiments.py"),
                                 "all", "--no-figures", "--dry-run"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pdflatex", result.stdout)
        self.assertNotIn("paper_update/build_figure.py", result.stdout)

    def test_tex_preflight_precedes_expensive_experiments(self):
        with patch.object(sys, "argv", ["run_experiments.py", "all"]), \
             patch.object(launcher, "check_environment", side_effect=SystemExit("missing TeX")) as check, \
             patch.object(launcher, "execute") as execute:
            with self.assertRaisesRegex(SystemExit, "missing TeX"):
                launcher.main()
            check.assert_called_once_with(require_tex=True)
            execute.assert_not_called()

    def test_missing_prerequisites_explain_required_stages(self):
        with patch.object(Path, "is_file", return_value=False):
            for stage, producer in [("lr", "core"), ("prediction", "lr"), ("figures", "head")]:
                with self.assertRaises(SystemExit) as context:
                    launcher.check_prerequisites(stage)
                self.assertIn(producer, str(context.exception))
                self.assertIn("Run the required stages first", str(context.exception))

    def test_direct_evaluator_does_not_load_historical_manifests_on_startup(self):
        script = ROOT / "experiments/nonlinear_allocation_2026_10_08/independent_evaluation.py"
        with tempfile.TemporaryDirectory(prefix="dp_sgd_evaluator_help_") as directory:
            result = subprocess.run([sys.executable, "-B", str(script), "--help"],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cohort", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
