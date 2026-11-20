#!/usr/bin/env python3
"""Run every chapter script and compare its stdout with the output printed in the book.

Usage:
  python3 tools/verify.py                 # all chapters
  python3 tools/verify.py 01-matmul 07    # some chapters, by directory name or number

Each NN-*/exKK.py runs in a fresh temporary directory (300 s limit) with
PYTHONHASHSEED=0 and OMP_NUM_THREADS=1. It passes when stdout equals exKK.out.txt,
ignoring trailing whitespace. Exit code 1 if any script fails.

The expected outputs were produced on macOS arm64, Python 3.11, torch 2.15.0 (CPU),
numpy 2.4.6. Other platforms or versions can differ in the last printed digit of
floating-point results; such a diff is reported, not hidden.
"""
import glob
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(s):
    return '\n'.join(line.rstrip() for line in s.strip('\n').splitlines())


def chapters(args):
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, '[0-3][0-9]-*')) if os.path.isdir(d))
    if not args:
        return dirs
    return [d for d in dirs if any(os.path.basename(d) == a or os.path.basename(d).startswith(a + '-') for a in args)]


def run(script):
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1'}
        try:
            r = subprocess.run([sys.executable, script], cwd=tmp, capture_output=True, text=True,
                               timeout=300, env=env)
        except subprocess.TimeoutExpired:
            return None, 'timeout after 300 s'
        if r.returncode != 0:
            tail = r.stderr.strip().splitlines()[-1:] or ['(no stderr)']
            return None, f'exit {r.returncode}: {tail[0]}'
        return r.stdout, None


def main():
    failed = total = 0
    for d in chapters(sys.argv[1:]):
        for script in sorted(glob.glob(os.path.join(d, 'ex[0-9][0-9].py'))):
            total += 1
            rel = os.path.relpath(script, ROOT)
            out, err = run(script)
            want = open(script[:-3] + '.out.txt', encoding='utf-8').read()
            if err:
                failed += 1
                print(f'FAIL {rel}: {err}')
                continue
            a, b = norm(want).splitlines(), norm(out).splitlines()
            if a == b:
                print(f'PASS {rel}')
                continue
            failed += 1
            i = next(i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i])
            print(f'FAIL {rel}: line {i + 1}: expected {a[i] if i < len(a) else "<none>"!r}, got {b[i] if i < len(b) else "<none>"!r}')
    print(f'{total - failed}/{total} scripts match the book')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
