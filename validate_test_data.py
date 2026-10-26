from pathlib import Path
from PIL import Image

root = Path("datasets/test_detection")
images = root / "images"
labels = root / "labels"

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_map = {p.stem: p for p in images.iterdir()
             if p.is_file() and p.suffix.lower() in image_exts}
label_map = {p.stem: p for p in labels.glob("*.txt")}

missing_labels = sorted(set(image_map) - set(label_map))
missing_images = sorted(set(label_map) - set(image_map))
invalid = []
total_objects = 0

for stem, path in label_map.items():
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            vals = list(map(float, parts))
            if len(vals) != 5:
                invalid.append((path.name, line_no, "not 5 values"))
                continue
            cls, x, y, w, h = vals
            if cls != int(cls) or not 0 <= cls < 8:
                invalid.append((path.name, line_no, "class ID outside 0-7"))
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                invalid.append((path.name, line_no, "coordinate outside 0-1"))
            if w <= 0 or h <= 0 or x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
                invalid.append((path.name, line_no, "invalid box geometry"))
            total_objects += 1
        except Exception as e:
            invalid.append((path.name, line_no, str(e)))

print("Images:", len(image_map))
print("Label files:", len(label_map))
print("Annotations:", total_objects)
print("Images without labels:", len(missing_labels))
print("Labels without images:", len(missing_images))
print("Invalid annotation lines:", len(invalid))
for item in invalid[:20]:
    print("INVALID:", item)
for name in missing_labels[:10]:
    print("MISSING LABEL:", name)
for name in missing_images[:10]:
    print("MISSING IMAGE:", name)
