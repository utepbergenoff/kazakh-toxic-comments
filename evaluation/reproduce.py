"""Recompute the reported metrics from text-free saved predictions. No API calls or training."""
from pathlib import Path
import contextlib,hashlib,io,json,runpy
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
for name,expected in manifest.items():
    actual=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert actual==expected, f'Changed or corrupted input: {name}'
print(f'Input checksums verified: {len(manifest)} files')
log=io.StringIO()
with contextlib.redirect_stdout(log):
    for check in ['metrics.py','per_label.py','bootstrap.py']:
        runpy.run_path(str(ROOT/'checks'/check),run_name='__main__')
report=log.getvalue()+'\nPASS: all recorded metrics and six bootstrap contrasts reproduced.\n'
(ROOT/'reproduction_report.txt').write_text(report)
print(report)
