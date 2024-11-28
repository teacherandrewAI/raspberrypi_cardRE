import cv2

# Intenta abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    print("Cámara abierta correctamente. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        break

    cv2.imshow("Vista de la Cámara", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
