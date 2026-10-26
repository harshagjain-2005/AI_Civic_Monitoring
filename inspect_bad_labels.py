from pathlib import Path

folder = Path("datasets/test_detection/labels")
bad_files = [
    "b27_jpeg.rf.f7840b7243a3405c7b86d4be80392876.txt",
    "IMG20231024165942_jpg.rf.50afaf513a0e4c0583f4c91955b0fe13.txt",
    "IMG20231024171422_jpg.rf.8a26217c54d6bf5536818f00593018a2.txt",
    "Plastic_Bottle-10557-_jpg.rf.2ec6b2723ccee5d10b859d28b02f4dab.txt",
    "Plastic_Bottle-10713-_jpg.rf.c9791ac05bbee61d505f9c22c57e88e9.txt",
    "Plastic_Bottle-2488-_jpg.rf.9aa5a14a8eae2e3e3d1cc4e616cbdcde.txt",
    "WhatsApp-Image-2023-10-23-at-19-07-13-1-_jpeg.rf.c3999032ac9295ce3f42b01779c131b3.txt",
    "WhatsApp-Image-2023-10-23-at-19-18-04_jpeg.rf.149933189aefea60d504d239f96fcb59.txt",
    "WhatsApp-Image-2023-10-23-at-19-18-14_jpeg.rf.743478a930b9962d272b5ae6eb966a4a.txt",
]

for name in bad_files:
    path = folder / name
    if not path.exists():
        print("\nMISSING FILE:", name)
        continue

    print("\n---", name, "---")
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)
        bounds = (x-w/2, y-h/2, x+w/2, y+h/2)
        if bounds[0] < -1e-6 or bounds[1] < -1e-6 or bounds[2] > 1+1e-6 or bounds[3] > 1+1e-6 or w <= 0 or h <= 0:
            print("Line:", line_no)
            print("Original:", line)
            print("Box bounds (left, top, right, bottom):", bounds)
