import cv2
from ultralytics import YOLO


def main():

	model = YOLO("../models/trained/vehicle_detector.pt")

	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()

		if not ret:
			break

		results = model(frame)

		annotated = results[0].plot()

		cv2.imshow("Vehicle Detection", annotated)

		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
