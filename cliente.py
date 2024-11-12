import cv2
import base64
import requests
import time

# URL do serviço Flask
URL = "http://127.0.0.1:5000/predict"

def send_frame_to_flask(frame):
    # Converter o frame para JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Converter o frame para base64
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    # Enviar a imagem para a API Flask
    response = requests.post(URL, json={"image": frame_base64})
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predição: {data['prediction_pt']} / {data['prediction_en']}")
    else:
        print("Erro na previsão:", response.json().get("error"))

def main():
    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(0)
    # Definir o intervalo de envio (em segundos)
    interval = 1  # 1 frame por segundo
    last_sent = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar a imagem")
            break

        # Exibir o frame na tela
        cv2.imshow("Camera", frame)

        # Enviar o frame para a API a cada `interval` segundos
        if time.time() - last_sent > interval:
            send_frame_to_flask(frame)
            last_sent = time.time()

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
