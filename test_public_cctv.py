import cv2
import time

STREAM_URL = "https://media-sfs5.vdotcameras.com/rtplive/NO0328/playlist_sfm4s.m3u8"

print("Opening public CCTV stream...")

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Could not open public CCTV stream.")
    exit()

print("SUCCESS: Public CCTV stream opened.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

for i in range(20):
    print(f"Reading frame {i + 1}...")

    start = time.time()
    ret, frame = cap.read()
    elapsed = time.time() - start

    print(f"ret={ret}, time={elapsed:.2f}s")

    if ret and frame is not None:
        print(f"Frame received: {frame.shape}")

        cv2.imshow("CivicGuard - Public CCTV", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("No frame received.")

cap.release()
cv2.destroyAllWindows()