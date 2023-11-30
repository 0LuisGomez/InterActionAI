import csv
import copy
import itertools
from collections import Counter
from collections import deque

import os
import cv2 as cv
import numpy as np
import mediapipe as mp

from utilities import CvFpsCalc
from model import ClasificarPuntos
from model import ClasificadorHistoria
from model import ObjectDetector

landmark_color = (237, 254, 0)  # Hex #00feed -> BGR
connection_color = (34, 254, 38)  # Hex #fe89e5 -> BGR
circle_color = (233, 76, 119)  # Hex #fe89e5 -> BGR

all_rects = []

interaction_data = None

def main():
    global interaction_data
    ####  Declarar variables relevantes para el programa #####

    cap_device = 0
    cap_width = 960
    cap_height = 540

    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    use_brect = True

    ####  iniciar la camara ####
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #### iniciar mediapipe hand solutions para la deteccion de los puntos clave ####
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    model_path = os.path.join('.', 'model', 'Clasificador_objetos', 'best.pt')
    detector = ObjectDetector(model_path)

    current_detections = []

    screen_grid = None # Variable to store the grid

    clasificador_puntos = ClasificarPuntos()

    Clasificador_historia = ClasificadorHistoria()

    #### csv para la traduccion de la respuesta de la red (0-2) hacia valores string (open, close, pointed) ####
    with open(
        "model\Clasificador_de_puntos\clasificador_puntos_label.csv",
        encoding="utf-8-sig",
    ) as f:
        clasificador_puntos_labels = csv.reader(f)
        clasificador_puntos_labels = [row[0] for row in clasificador_puntos_labels]
    with open(
        "model\Clasificador_historia_puntos\Clasificador_historia_label.csv",
        encoding="utf-8-sig",
    ) as f:
        clasificador_historia_label = csv.reader(f)
        clasificador_historia_label = [row[0] for row in clasificador_historia_label]

    #### uso de una calculadora de fps para observar el rendimiento segun las redes ####
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #### declarando variables para la historia de los puntos ####
    history_length = 16
    point_history = deque(maxlen=history_length)

    #### Declarando variables para la historia de los gestos ####
    finger_gesture_history = deque(maxlen=history_length)

    # inicio  ########################################################################

    while True:
        fps = cvFpsCalc.get()

        #### asignar esc como tecla salida ####
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        #### recibir la imagen, voltearla y crear una copia ####
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # voltear imagen (asi se entrenaron las redes) 
        debug_image = copy.deepcopy(image)


        #### convertir la imagen a rgb para pasarla por mediapipe ####
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        BUTTON_INTERACTION_THRESHOLD = (180, 220)
        SCREEN_INTERACTION_THRESHOLD = 150
        TOO_CLOSE_THRESHOLD = 515

        key = cv.waitKey(1)
        if key == ord('d'):  # Press 'd' to detect
            current_detections = detector.detect(debug_image)
            for det in current_detections:
                if det[4] == 'pantalla':  # If screen is detected
                    x1, y1, x2, y2 = map(int, det[:4])
                    screen_grid = create_grid(x1, y1, x2, y2)
                    break  # Assuming only one screen


        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                #### se crea la caja en base a las medidas y se extraen los landmarks y se enlistan para su procesamiento ####
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                hand_size = calculate_hand_size(brect)

                # calculando los landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                #### se continua con el procesamiento de los landmarks para la linealizacion de datos ####
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history
                )

                #### se pasan por la funcion de la red neuronal clasificadora de puntos #####
                hand_sign_id = clasificador_puntos(pre_processed_landmark_list)
                if hand_sign_id == 2:  # apuntando
                    point_history.append(landmark_list[8])  # coordenadas de la punta del index
                else:
                    point_history.append([0, 0])

                #### Se pasan por la funcion de la red neuronal clasificadora de historia ####
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = Clasificador_historia(
                        pre_processed_point_history_list
                    )

                #### se utiliza la variable que creamos de la historia para enlistar los mas recientes y determinar por comun ####
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                #### una vez que tenemos los datos solo los pasamos por las funciones de dibujo para visualizacion ####
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    clasificador_puntos_labels[hand_sign_id],
                    clasificador_historia_label[most_common_fg_id[0][0]],
                )

                if screen_grid and point_history:
                    # Use the last point in point_history for the fingertip position
                    fingertip_x, fingertip_y = point_history[-1]

                if hand_size < SCREEN_INTERACTION_THRESHOLD:
                    if screen_grid is not None:  # Check if screen_grid is not None
                        for i, cell in enumerate(screen_grid):
                            cell_x1, cell_y1, cell_x2, cell_y2 = cell
                            if cell_x1 <= fingertip_x <= cell_x2 and cell_y1 <= fingertip_y <= cell_y2:
                                print(f"Fingertip is inside grid cell {i}")
                                # Perform the action associated with this cell
                                break

                elif BUTTON_INTERACTION_THRESHOLD[0] <= hand_size <= BUTTON_INTERACTION_THRESHOLD[1]:
                    for det in current_detections:
                        x1, y1, x2, y2, class_name, _ = det
                        if "boton" in class_name:  # Check only for button detections
                            if x1 <= fingertip_x <= x2 and y1 <= fingertip_y <= y2:
                                print(f"Interacting with {class_name}")
                                # Perform the action associated with this button
                                break  # Uncomment if you only want one button interaction at a time
                elif hand_size > TOO_CLOSE_THRESHOLD:
                    print("Hand too close")
                    # Logic for when the hand is too close

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps)

        for x1, y1, x2, y2, class_name, bbox_color in current_detections:
            cv.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 1)
            cv.putText(debug_image, class_name.upper(), (int(x1), int(y1 - 10)), cv.FONT_HERSHEY_PLAIN, 1.0, bbox_color, 3, cv.LINE_AA)

        
        if screen_grid:
            for cell in screen_grid:
                cv.rectangle(debug_image, (int(cell[0]), int(cell[1])), (int(cell[2]), int(cell[3])), (0, 255, 0), 1)  # Draw grid cells

        #### iniciamos la ventana ####
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calculate_hand_size(brect):
    # Calculate the diagonal length of the bounding rectangle
    diagonal_length = np.sqrt((brect[2] - brect[0])**2 + (brect[3] - brect[1])**2)
    return diagonal_length

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # creamos la lista de landmarks para x y y, z escala diferente asi que no se usa para la red ####
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # aqui se realiza la parte del procesamiento donde se consigue la posicion relativa conforme al punto 0 ####
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # aqui estamos haciendo el "flatten" de la lista (convertirla de un array multi-dimensional a una sola lista larga)
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # determinar el valor mas grande para normalizar en base a el
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    ## Se realiza una enumeracion similar a la vista en la funcion previa para los datos de la red clasificadora de historia
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    ## se vuelve aplicar el "flatten" para convertirlo en una lista de 1D
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def draw_landmarks(image, landmark_point):
    # asegurarse de que se detecte la mano
    if len(landmark_point) > 0:
        # pulgar
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), connection_color, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), connection_color, 2)

        # index
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), connection_color, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), connection_color, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), connection_color, 2)

        # middle
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]),connection_color,2
        )

        # ring
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), connection_color,2
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), connection_color, 2
        )

        # pinkie
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), connection_color,2
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), connection_color, 2
        )

        # palma
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), connection_color, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), connection_color,2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), connection_color, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), connection_color, 2)
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), connection_color, 2
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), connection_color, 2
        )

    # puntos clave
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4: 
            cv.circle(image, (landmark[0], landmark[1]), 8, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8: 
            cv.circle(image, (landmark[0], landmark[1]), 8, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12: 
            cv.circle(image, (landmark[0], landmark[1]), 8, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16: 
            cv.circle(image, (landmark[0], landmark[1]), 8, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19: 
            cv.circle(image, (landmark[0], landmark[1]), 5, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20: 
            cv.circle(image, (landmark[0], landmark[1]), 8, landmark_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # creacion de la caja si el brect es true
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    # dibujar los circulos cuando se apunta en base a la historia
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 + int(index / 2), circle_color, 2
            )

    return image


def draw_info(image, fps):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    return image

def create_grid(x1, y1, x2, y2, rows=12, cols=12):
    grid = []
    width, height = x2 - x1, y2 - y1
    cell_width, cell_height = width / cols, height / rows  # Cell dimensions for 32x32 grid

    for i in range(rows):
        for j in range(cols):
            # Calculate the coordinates for each cell
            cell_x1 = x1 + j * cell_width
            cell_y1 = y1 + i * cell_height
            cell_x2 = cell_x1 + cell_width
            cell_y2 = cell_y1 + cell_height

            # Adjust if cell extends beyond the screen area
            cell_x2 = min(cell_x2, x2)
            cell_y2 = min(cell_y2, y2)

            cell = (cell_x1, cell_y1, cell_x2, cell_y2)
            grid.append(cell)
    return grid


def is_point_in_cell(x, y, cell):
    x1, y1, x2, y2 = cell
    return x1 <= x <= x2 and y1 <= y <= y2


if __name__ == "__main__":
    main()
