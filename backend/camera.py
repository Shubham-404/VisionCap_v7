import cv2

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("❌ Failed to open camera")
    exit()

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
