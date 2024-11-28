from picamera2 import Picamera2
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

# Rutas a los archivos del modelo y etiquetas
MODEL_PATH = "model/vww_96_grayscale_quantized.tflite"
LABELS_PATH = "model/labels.txt"

# Cargar etiquetas
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Cargar el modelo de TensorFlow Lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dimensiones de entrada del modelo
input_shape = input_details[0]["shape"]

# Función para preprocesar la imagen capturada
def preprocess_image(image):
    """
    Preprocesa una imagen para ajustarse al modelo TensorFlow Lite.
    - Cambia el tamaño de la imagen.
    - Convierte la imagen a escala de grises (si el modelo lo requiere).
    - Escala los valores de los píxeles al rango [0, 1].
    """
    image = cv2.resize(image, (input_shape[1], input_shape[2]))  # Cambiar tamaño
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    image = np.expand_dims(image, axis=-1)  # Añadir canal de profundidad
    image = np.expand_dims(image, axis=0)  # Añadir dimensión batch
    image = image.astype(np.float32) / 255.0  # Normalizar al rango [0, 1]
    return image

# Configurar la cámara
picam2 = Picamera2()
picam2.start()

# Función principal para captura y reconocimiento
def predict():
    """
    Captura imágenes desde la cámara, realiza predicciones con el modelo
    y muestra los resultados en una ventana en tiempo real.
    """
    print("Presiona 'q' para salir.")
    
    while True:
        frame = picam2.capture_array()  # Capturar imagen en tiempo real

        # Preprocesar la imagen capturada
        input_data = preprocess_image(frame)

        # Realizar la predicción
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Obtener los resultados de salida del modelo
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = labels[np.argmax(output_data)]  # Etiqueta con mayor probabilidad
        confidence = np.max(output_data)  # Confianza asociada a la predicción

        # Mostrar resultados en la imagen
        text = f"Predicción: {predicted_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen en una ventana
        cv2.imshow("Reconocimiento en Tiempo Real", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ejecutar la función de predicción
if __name__ == "__main__":
    predict()
    picam2.stop()
