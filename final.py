import os
import cv2 as cv
import numpy as np
import time
import webcam
import kalman
from enum import Enum

def load_set(name: str) -> list: 
    """
    Carrega imagens de um conjunto de dados.
    """
    extensions = [".jpg", ".jpeg", ".png"]
    images = []
    dataset_folder = "dataset"
    dataset_path = os.path.join(dataset_folder, name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Conjunto de dados "{name}" não encontrado.')
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f'"{dataset_path}" não é um diretório.')
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in extensions:
                images.append(os.path.join(root, file))
    images.sort()
    return images


def load_image(path: str) -> np.ndarray:
    """
    Carrega uma imagem.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Imagem "{path}" não encontrada.')
    image = cv.imread(path)
    return image

class FeatureExtractMode(Enum):
    ORB1 = "orb1"
    ORB2 = "orb2"
    HARRIS1 = "harris1"
    HARRIS2 = "harris2"
    SHI_TOMASI1 = "shi-tomasi1"
    SHI_TOMASI2 = "shi-tomasi2"

def extract_features(image: np.ndarray, points_to_track: int = 10, mode: FeatureExtractMode = FeatureExtractMode.ORB1) -> np.ndarray:
    """
    Extrai características de uma imagem.
    """
    corners = []
    if mode == FeatureExtractMode.ORB1:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = cv.ORB_create(nfeatures=points_to_track)
        kp, des = kp.detectAndCompute(gray, None)
        corners = np.array([k.pt for k in kp], dtype=np.float32)
    elif mode == FeatureExtractMode.ORB2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = cv.ORB_create(nfeatures=points_to_track, scoreType=cv.ORB_FAST_SCORE)
        kp, des = kp.detectAndCompute(gray, None)
        corners = np.array([k.pt for k in kp], dtype=np.float32)
    elif mode == FeatureExtractMode.HARRIS1:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tracked_points_number = points_to_track
        corners = cv.goodFeaturesToTrack(gray, tracked_points_number, 0.01, 10, useHarrisDetector=True)
        corners = corners.reshape(-1, 2) # Remove [[x, y]] to [x, y]
    elif mode == FeatureExtractMode.HARRIS2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tracked_points_number = points_to_track
        corners = cv.goodFeaturesToTrack(gray, tracked_points_number, 0.02, 4, useHarrisDetector=True, k=0.04)
        corners = corners.reshape(-1, 2)
    elif mode == FeatureExtractMode.SHI_TOMASI1:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tracked_points_number = points_to_track
        corners = cv.goodFeaturesToTrack(gray, tracked_points_number, 0.01, 10, useHarrisDetector=False)
        corners = corners.reshape(-1, 2) # Remove [[x, y]] to [x, y]
    elif mode == FeatureExtractMode.SHI_TOMASI2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tracked_points_number = points_to_track
        corners = cv.goodFeaturesToTrack(gray, tracked_points_number, 0.02, 4, useHarrisDetector=False, k=0.04)
        corners = corners.reshape(-1, 2)
    return corners


def draw_features(image: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Desenha características em uma imagem.
    """
    for feature in features:
        x, y = feature.ravel()
        center = (int(x), int(y))
        radius = 3
        color = (0, 255, 0)
        cv.circle(image, center, radius, color, -1)
    return image


def take_pair_of_images(webcam: webcam.Webcam, sleeptime: float = 1/60) -> tuple:
    """
    Captura um par de imagens da webcam.
    """
    image1 = webcam_device.take_image()
    time.sleep(sleeptime)
    image2 = webcam_device.take_image()
    return image1, image2


def get_movement_tracks(
    image1: np.ndarray, features1: np.ndarray, image2: np.ndarray, features2: np.ndarray, smooth_tracks: bool = True
) -> np.ndarray:
    """
    Rastreia características em uma imagem.
    """

    def smooth_tracks(tracks: np.ndarray) -> np.ndarray:
        """
        Suaviza rastros de características.
        Divide a imagem em segmentos de 16x16 pixels e agrupa os pontos em segmentos.
        Após agrupar os pontos, calcula a média da direção do movimento.
        Recalcula-se a posição dos pontos baseado na média da direção do movimento.
        """
        segment_size_px = 24
        # pseudocode
        # criar lista de segmentos vazia
        # para cada ponto em tracks
        #   segmento = ponto // segment_size_px
        #   adicionar ponto a lista de segmentos[segmento]
        # para cada segmento em lista de segmentos
        #   calcular média da direção do movimento
        #   para cada ponto em segmento
        #       recalcular posição do ponto baseado na média da direção do movimento
        #       adicionar ponto a lista de pontos recalculados
        # retornar lista de pontos recalculados
        
        segments = {}
        for track in tracks:
            x, y = track.ravel()
            segment_sector_x = int(x // segment_size_px)
            segment_sector_y = int(y // segment_size_px)
            segment = (segment_sector_x, segment_sector_y)
            if segment[0] not in segments:
                segments[segment[0]] = {}
            if segment[1] not in segments[segment[0]]:
                segments[segment[0]][segment[1]] = []
            segments[segment[0]][segment[1]].append(track)
        recalculated_tracks = []
        for segment_x in segments:
            for segment_y in segments[segment_x]:
                segment = segments[segment_x][segment_y]
                mean_direction = np.mean(np.linalg.norm(segment, axis=1))
                for track in segment:
                    x, y = track.ravel()
                    track_nd = np.array([x, y])
                    track_direction = np.linalg.norm([x, y])
                    recalculated_track = track_nd
                    recalculated_tracks.append(recalculated_track)
        return np.array(recalculated_tracks)
                

    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    features2, status, error = cv.calcOpticalFlowPyrLK(
        gray1, gray2, features1, features2
    )
    if smooth_tracks:
        return smooth_tracks(features2)
    return features2


def draw_tracks(
    image: np.ndarray, features1: np.ndarray, tracks: np.ndarray
) -> np.ndarray:
    """
    Desenha rastros de características em uma imagem.
    """
    min_speed_color = (255, 0, 0)
    max_speed_color = (0, 0, 255)
    max_speed_value = 100
    for feature, track in zip(features1, tracks):
        start_x, start_y = feature.ravel()
        end_x, end_y = track.ravel()
        speed = np.linalg.norm([end_x - start_x, end_y - start_y])
        color = (
            int(
                (1 - speed / max_speed_value) * min_speed_color[0]
                + speed / max_speed_value * max_speed_color[0]
            ),
            int(
                (1 - speed / max_speed_value) * min_speed_color[1]
                + speed / max_speed_value * max_speed_color[1]
            ),
            int(
                (1 - speed / max_speed_value) * min_speed_color[2]
                + speed / max_speed_value * max_speed_color[2]
            ),
        )
        size = 2
        cv.arrowedLine(
            image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, size
        )
    return image


def set_kalman_points(kalman_filter: kalman.Kalman, features: np.ndarray):
    """
    Atualiza os pontos do filtro de Kalman.
    """
    if len(features) != kalman_filter.size:
        if len(features) > kalman_filter.size:
            features = features[:kalman_filter.size]
    for index, point in enumerate(features):
        if index >= kalman_filter.size:
            break
        kf = kalman_filter.kalmans[index]
        x, y = point.ravel()
        if kf.statePre.all() == 0:
            kf.statePre = np.array([x, y], np.float32)
            kf.statePost = np.array([x, y], np.float32)
        kf.correct(np.array([[x], [y]], np.float32))


def predict_kalman_points(kalman_filter: kalman.Kalman) -> np.ndarray:
    """
    Obtém os pontos do filtro de Kalman.
    """
    value = np.zeros((kalman_filter.size, 2), np.float32)
    for index, k in enumerate(kalman_filter.kalmans):
        value[index] = k.predict()
    return value


def draw_kalman_points(image: np.ndarray, kalman: np.ndarray) -> np.ndarray:
    """
    Desenha os pontos do filtro de Kalman em uma imagem.
    """
    for index, point in enumerate(kalman):
        x, y = point.ravel()
        center = (int(x), int(y))
        radius = 5
        color = (255, 0, 255)
        cv.circle(image, center, radius, color, -1)
        cv.putText(
            image,
            f"{index + 1}",
            center,
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )


def process_set(set_name: str, points_to_track: int, detector_mode: FeatureExtractMode, smooth_tracks: bool, enable_experimental_feature_ordering: bool) -> np.ndarray:
    """
    Processa um conjunto de dados.
    """
    def draw_kalman_history(image: np.ndarray, history: list) -> np.ndarray:
        """
        Desenha o histórico do filtro de Kalman em uma imagem.
        """
        previous_historyitem = None
        for historyindex, historyitem in enumerate(history):
            historyitem: np.ndarray = historyitem
            for index, point in enumerate(historyitem):
                # Draw line
                if previous_historyitem is not None:
                    previous_point = previous_historyitem[index]
                    x1, y1 = previous_point.ravel()
                    x2, y2 = point.ravel()
                    color = (255, 100, 100, 100)
                    cv.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # Draw circle point
                x, y = point.ravel()
                center = (int(x), int(y))
                radius = 3
                color = (255, 100, 100, 100)
                cv.circle(image, center, radius, color, -1)
                cv.putText(
                    image,
                    f"{index + 1}",
                    center,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 100, 100),
                    1,
                    cv.LINE_AA,
                )
                # previous = point
            previous_historyitem = historyitem
        return image

    def draw_track_history(image: np.ndarray, history: list) -> np.ndarray:
        """
        Desenha o histórico de rastros em uma imagem.
        """
        previous_historyitem = None
        for historyindex, historyitem in enumerate(history):
            historyitem: np.ndarray = historyitem
            for index, point in enumerate(historyitem):
                # Draw line
                if previous_historyitem is not None:
                    previous_point = previous_historyitem[index]
                    x1, y1 = previous_point.ravel()
                    x2, y2 = point.ravel()
                    color = (100, 100, 255)
                    cv.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # Draw circle point
                x, y = point.ravel()
                center = (int(x), int(y))
                radius = 3
                color = (100, 100, 255)
                cv.circle(image, center, radius, color, -1)
                cv.putText(
                    image,
                    f"{index + 1}",
                    center,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (100, 100, 255),
                    1,
                    cv.LINE_AA,
                )
                # previous = point
            previous_historyitem = historyitem
        return image

    images = load_set(set_name)
    full_image_parts = []
    kalman_filter = kalman.Kalman(points_to_track)
    kalman_history = []
    track_history = []
    experimental_ordering_previous_points = None
    for i in range(len(images) - 1):
        image1 = load_image(images[i])
        image2 = load_image(images[i + 1])
        features1 = extract_features(image1, points_to_track, detector_mode)
        features2 = extract_features(image2, points_to_track, detector_mode)
        if enable_experimental_feature_ordering:
            if experimental_ordering_previous_points is not None:
                features1 = align_features_to_previous(experimental_ordering_previous_points, features1)
                features2 = align_features_to_previous(experimental_ordering_previous_points, features2)
        set_kalman_points(kalman_filter, features1)
        tracked_features = get_movement_tracks(image1, features1, image2, features2, smooth_tracks)
        if enable_experimental_feature_ordering:
            experimental_ordering_previous_points = tracked_features
        track_history.append(tracked_features)
        kalman_prediction = predict_kalman_points(kalman_filter)
        kalman_history.append(kalman_prediction)
        shown_image = image1
        shown_image = draw_features(shown_image, features1)
        shown_image = draw_tracks(shown_image, features1, tracked_features)
        draw_kalman_points(shown_image, kalman_prediction)
        full_image_parts.append(shown_image.copy())
    
    # Draw Kalman and Track History
    last_image = load_image(images[-1])
    draw_track_history(last_image, track_history)
    draw_kalman_history(last_image, kalman_history)
    full_image_parts.append(last_image)
    
    full_image = np.concatenate(full_image_parts, axis=1)
    return full_image

def align_features_to_previous(previous_points: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Alinha a ordem das características a um conjunto de características anterior.
    O algoritmo tenta encontrar a característica mais próxima em relação a um conjunto de características anterior,
    mas evitando repetições de índices, evitando que uma característica fique no mesmo ponto ou localização que outra.
    """
    aligned_features = np.zeros_like(features)
    taken_indexes = np.zeros(len(previous_points), dtype=np.int32)
    if len(previous_points) != len(features):
        print(f"Previous points and features have different lengths: {len(previous_points)} != {len(features)}")
        return features
    for index, point in enumerate(previous_points):
        if index >= len(taken_indexes):
            break
        if index >= len(features):
            break
        if index >= len(aligned_features):
            break
        distance = np.linalg.norm(features - point, axis=1)
        ordered_indexes = np.argsort(distance)
        closest = None
        for ordered_index in ordered_indexes:
            if taken_indexes[ordered_index] == 0:
                closest = ordered_index
                taken_indexes[ordered_index] = 1
                break
        aligned_features[index] = features[closest]
    return aligned_features

if __name__ == "__main__":   
    # Point Count
    point_count_list = [2, 5, 10, 20, 50, 100] # define a quantidade de pontos a serem rastreados
    points_to_track_index = 1
    if points_to_track_index >= len(point_count_list): # se a lista terminar, reinicia
        points_to_track_index = 0
    points_to_track = point_count_list[points_to_track_index]
    
    # Sets
    available_sets = [f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))] # define os conjuntos de dados disponíveis
    available_sets.sort()
    set_index = 0
    if set_index >= len(available_sets):
        set_index = 0
    if len(available_sets) == 0:
        set_index = None
        
    # Detectors
    detectors = [f for f in FeatureExtractMode] # define os detectores de características disponíveis
    detector_index = 0
    detector_mode = detectors[detector_index]
    if detector_index >= len(detectors):
        detector_index = 0
    
    
    webcam_device = webcam.Webcam() # inicializa a webcam
    kalman_filter = kalman.Kalman(points_to_track) # inicializa o filtro de Kalman
    
    show_features = True # define se as características devem ser exibidas
    smooth_tracks = True # define se os rastros devem ser suavizados
    show_tracks = True # define se os rastros devem ser exibidos
    show_kalman = True # define se os pontos do filtro de Kalman devem ser exibidos
    show_hud = True # define se o HUD deve ser exibido
    enable_experimental_feature_ordering = False # define se a ordenação experimental de características deve ser ativada
    experimental_ordering_previous_points = None # define os pontos anteriores para a ordenação experimental

    while True: # loop principal
        image_pair = take_pair_of_images(webcam_device) # captura um par de imagens
        features1 = extract_features(image_pair[0], points_to_track, detector_mode) # extrai características da primeira imagem
        features2 = extract_features(image_pair[1], points_to_track, detector_mode) # extrai características da segunda imagem

        if enable_experimental_feature_ordering: # se a ordenação experimental de características estiver ativada
            if experimental_ordering_previous_points is not None: # se houver pontos anteriores
                features1 = align_features_to_previous(experimental_ordering_previous_points, features1) # alinha as características da primeira imagem
                features2 = align_features_to_previous(experimental_ordering_previous_points, features2) # alinha as características da segunda imagem
        set_kalman_points( # atualiza os pontos do filtro de Kalman
            kalman_filter, features1
        )  # update points to kalman filter and get the predicted points
        tracked_features = get_movement_tracks(
            image_pair[0], features1, image_pair[1], features2, smooth_tracks
        )
        if enable_experimental_feature_ordering: # se a ordenação experimental de características estiver ativada
            experimental_ordering_previous_points = tracked_features # define os pontos anteriores para a ordenação experimental
        kalman_prediction = predict_kalman_points(kalman_filter) # obtém os pontos do filtro de Kalman
        shown_image = image_pair[0]

        if show_features: # se as características devem ser exibidas
            shown_image = draw_features(image_pair[0], features1)
        if show_tracks: # se os rastros devem ser exibidos
            draw_tracks(shown_image, features1, tracked_features)
        if show_kalman: # se os pontos do filtro de Kalman devem ser exibidos
            draw_kalman_points(shown_image, kalman_prediction)
        if show_hud: # se o HUD deve ser exibido
            cv.putText(
                shown_image,
                f"Features (f): {show_features}",
                (10, 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Detector (m): {detector_mode.value}",
                (10, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 200),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Points to track (c): {points_to_track}",
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Flow Tracks (t): {show_tracks}",
                (10, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Smooth Flow Tracks (y): {smooth_tracks}",
                (10, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Kalman prediction (k): {show_kalman}",
                (10, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"HUD (h): {show_hud}",
                (10, 140),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Press 'q' to quit.",
                (10, 160),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Set selection (s): {set_index + 1}",
                (10, 180),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Process set (p): {available_sets[set_index]}",
                (10, 200),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Experimental Ordering (e): {enable_experimental_feature_ordering}",
                (10, 220),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 255),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                shown_image,
                f"Livecam Mode",
                (shown_image.shape[1] - 180, 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv.LINE_AA,
                False,
            )
            

        cv.imshow("Image", shown_image) # exibe a imagem
        delay = int(1000 / 60)
        key = cv.waitKey(delay)

        if key & 0xFF == ord("f"): # se a tecla 'f' for pressionada
            show_features = not show_features # inverte o estado de exibição das características
        if key & 0xFF == ord("t"): # se a tecla 't' for pressionada
            show_tracks = not show_tracks # inverte o estado de exibição dos rastros
        if key & 0xFF == ord("y"): # se a tecla 'y' for pressionada
            smooth_tracks = not smooth_tracks  # inverte o estado de suavização dos rastros
        if key & 0xFF == ord("k"): # se a tecla 'k' for pressionada
            show_kalman = not show_kalman # inverte o estado de exibição dos pontos do filtro de Kalman
        if key & 0xFF == ord("h"): # se a tecla 'h' for pressionada
            show_hud = not show_hud # inverte o estado de exibição do HUD
        if key & 0xFF == ord("m"): # se a tecla 'm' for pressionada
            detector_index += 1
            if detector_index >= len(detectors):
                detector_index = 0
            detector_mode = detectors[detector_index]
            experimental_ordering_previous_points = None
        if key & 0xFF == ord("c"): # se a tecla 'c' for pressionada
            points_to_track_index += 1
            if points_to_track_index >= len(point_count_list):
                points_to_track_index = 0
            points_to_track = point_count_list[points_to_track_index]
            experimental_ordering_previous_points = None
            kalman_filter = kalman.Kalman(points_to_track)
        if key & 0xFF == ord("s"): # se a tecla 's' for pressionada
            set_index += 1
            if set_index >= len(available_sets):
                set_index = 0
        if key & 0xFF == ord("p"):  # se a tecla 'p' for pressionada
            setname = available_sets[set_index]
            full_image = process_set(setname, points_to_track, detector_mode, smooth_tracks, enable_experimental_feature_ordering)
            cv.imwrite(f"{setname}_processed.png", full_image)
        if key & 0xFF == ord("e"):  # se a tecla 'e' for pressionada
            enable_experimental_feature_ordering = not enable_experimental_feature_ordering
            experimental_ordering_previous_points = None
        if key & 0xFF == ord("q"): # se a tecla 'q' for pressionada
            cv.destroyAllWindows()
            break
    del webcam_device
