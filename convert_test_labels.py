from pathlib import Path

source = Path("datasets/test/labels")
destination = Path("datasets/test_detection/labels")
destination.mkdir(parents=True, exist_ok=True)

converted_boxes = 0
copied_boxes = 0

for file in source.glob("*.txt"):
    output_lines = []

    for line in file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue

        values = list(map(float, parts))
        class_id = int(values[0])

        if len(values) == 5:
            output_lines.append(line)
            copied_boxes += 1
            continue

        coords = values[1:]
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError(f"Unexpected format in {file.name}: {line}")

        xs = coords[0::2]
        ys = coords[1::2]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        output_lines.append(
            f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
        )
        converted_boxes += 1

    (destination / file.name).write_text(
        "\n".join(output_lines) + ("\n" if output_lines else "")
    )

print("Copied existing boxes:", copied_boxes)
print("Converted polygon annotations:", converted_boxes)
print("Output directory:", destination)
