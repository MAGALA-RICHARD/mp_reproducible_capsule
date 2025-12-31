import shutil
from pathlib import Path

dest = Path(r"D:\My_BOX\Box\MorrowPlotsRevisted2025\mp_reproducible_capsule")

dest.mkdir(parents=True, exist_ok=True)

for p in Path(__file__).parent.glob("*.py"):
    fi_name = dest/p.name
    shutil.copy(p, fi_name)
    print(fi_name, 'succeeded')
