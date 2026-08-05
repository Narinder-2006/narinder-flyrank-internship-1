import sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / 'work' / 'notebooks' / 'w04_baseline_score.ipynb'
if not NB_PATH.exists():
    print(f'Notebook not found: {NB_PATH}', file=sys.stderr)
    sys.exit(2)

nb = nbformat.read(str(NB_PATH), as_version=4)

# Create a globals dict similar to running as a script
globals_dict = {
    '__name__': '__main__',
    '__file__': str(NB_PATH),
}

for i, cell in enumerate(nb.cells, start=1):
    if cell.cell_type != 'code':
        continue
    source = cell.source
    if isinstance(source, list):
        source = ''.join(source)
    print(f"\n--- Executing cell {i} (len={len(source)} chars) ---")
    try:
        exec(compile(source, f'<cell {i}>', 'exec'), globals_dict)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f'Error executing cell {i}', file=sys.stderr)
        raise

print('\nExecution complete')
