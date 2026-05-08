"""Final verification: large-scale red detection after all fixes."""
import sys, subprocess
sys.path.insert(0, '/Users/stepanomelka/PyCharmMiscProject/SMT-deep')
from Generator.Utils.KernGenerator import KernGenerator

VENV_PYTHON = '/Users/stepanomelka/PyCharmMiscProject/SMT-deep/.venv/bin/python3'

def check_red_subprocess(kern_str):
    script = f'''
import sys, io, verovio
from contextlib import redirect_stdout, redirect_stderr
verovio.enableLog(verovio.LOG_OFF)
tk = verovio.toolkit()
kern = """{kern_str}"""
f = io.StringIO()
with redirect_stdout(f), redirect_stderr(f):
    tk.loadData(kern)
    svg = tk.renderToSVG(1)
if 'ff0000' in svg.lower():
    print("RED")
else:
    print("OK")
'''
    try:
        result = subprocess.run([VENV_PYTHON, '-c', script], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() == "RED"
    except:
        return None

gen = KernGenerator()

total = 200
red_count = 0
crash_count = 0

for i in range(total):
    measures = [2, 3, 4, 6, 8, 10][i % 6]
    kern = gen.generate(num_measures=measures)
    result = check_red_subprocess(kern)
    if result is None:
        crash_count += 1
    elif result:
        red_count += 1
        print(f"  [RED] Sample {i} ({measures} measures)")
        print(kern[:400])
        print("...\n")

print(f"\n{'='*60}")
print(f"FINAL RESULT: {red_count}/{total} samples had red elements")
print(f"Crashes: {crash_count}")
print(f"Clean: {total - red_count - crash_count}/{total}")
