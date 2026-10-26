from pathlib import Path

folder = Path("datasets/test_detection/labels")
tolerance = 0.0001
fixed = 0

for path in folder.glob("*.txt"):
    output = []

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Unexpected label format: {path.name}:{line_no}")

        cls, x, y, w, h = map(float, parts)
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        overflow = max(0, -left, -top, right - 1, bottom - 1)
        if overflow > tolerance:
            raise ValueError(
                f"Large overflow {overflow}: {path.name}:{line_no}"
            )

        new_left = max(0.0, left)
        new_top = max(0.0, top)
        new_right = min(1.0, right)
        new_bottom = min(1.0, bottom)

        if overflow > 0:
            x = (new_left + new_right) / 2
            y = (new_top + new_bottom) / 2
            w = new_right - new_left
            h = new_bottom - new_top
            fixed += 1

        output.append(f"{int(cls)} {x:.10f} {y:.10f} {w:.10f} {h:.10f}")

    path.write_text("\n".join(output) + ("\n" if output else ""))

print("Boundary-overflow annotations corrected:", fixed)
print("Evaluation-copy labels updated; original labels unchanged.")
