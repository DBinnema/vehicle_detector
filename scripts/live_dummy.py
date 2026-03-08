import cv2
import argparse
import time
import os

def main(source, width=None, height=None, record_path=None, screenshot_dir=None):
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    cap = cv2.VideoCapture(source)

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    if record_path:
        out = cv2.VideoWriter(record_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        start_time = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 1 / (time.time() - start_time)

        # Overlay text
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Resolution: {int(cap.get(3))}x{int(cap.get(4))}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show frame
        cv2.imshow('Live Video', frame)

        if out:
            out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Webcam index (int) or path/URL (string)')
    parser.add_argument('--width', type=int, help='Capture width')
    parser.add_argument('--height', type=int, help='Capture height')
    parser.add_argument('--record', type=str, help='Path to record video')
    parser.add_argument('--screenshot-dir', type=str, help='Directory to save screenshots')
    args = parser.parse_args()
    main(args.source, args.width, args.height, args.record, args.screenshot_dir)
