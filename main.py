import cv2 as cv

def extract_center_region():
    image = cv.imread('variant-8.jpg')
    if image is None:
        print("Не удалось загрузить изображение.")
        return

    h, w = image.shape[:2]
    crop_dim = 400
    x0 = w // 2 - crop_dim // 2
    y0 = h // 2 - crop_dim // 2
    cropped = image[y0:y0 + crop_dim, x0:x0 + crop_dim]

    cv.imwrite('cropped_image.jpg', cropped)
    print("Файл 'cropped_image.jpg' сохранён.")

    cv.imshow('Оригинал', image)
    cv.imshow('Обрезка', cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()


def track_largest_object(camera_index=2):
    capture = cv.VideoCapture(camera_index)
    target_size = (640, 480)

    while True:
        success, frame = capture.read()
        if not success:
            print("Не удалось получить кадр.")
            break

        frame = cv.resize(frame, target_size)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv.threshold(blurred, 110, 255, cv.THRESH_BINARY_INV)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv.contourArea)
            moment = cv.moments(contour)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])

                cv.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 255, 0), 2)
                cv.line(frame, (0, cy), (frame.shape[1], cy), (0, 255, 0), 2)
                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv.imshow('Видео', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


def follow_with_fly(camera_index=2):
    capture = cv.VideoCapture(camera_index)
    target_size = (640, 480)

    fly = cv.imread('fly64.png', cv.IMREAD_UNCHANGED)
    if fly is None:
        print("Файл 'fly64.png' не найден.")
        return

    if fly.shape[2] == 4:
        bgr = cv.cvtColor(fly, cv.COLOR_BGRA2BGR)
    else:
        bgr = fly
    fly_resized = cv.resize(bgr, (32, 32))

    while True:
        success, frame = capture.read()
        if not success:
            print("Ошибка захвата кадра.")
            break

        frame = cv.resize(frame, target_size)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv.threshold(blurred, 110, 255, cv.THRESH_BINARY_INV)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv.contourArea)
            moment = cv.moments(contour)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])

                cv.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 255, 0), 2)
                cv.line(frame, (0, cy), (frame.shape[1], cy), (0, 255, 0), 2)

                fh, fw = fly_resized.shape[:2]
                x1, y1 = cx - fw // 2, cy - fh // 2
                x2, y2 = x1 + fw, y1 + fh

                if 0 <= x1 and x2 <= frame.shape[1] and 0 <= y1 and y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = fly_resized

        cv.imshow('Fly Tracker', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    extract_center_region()
    # track_largest_object()
    # follow_with_fly()
