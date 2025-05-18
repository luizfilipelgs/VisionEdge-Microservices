import cv2
import numpy as np
import logging
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from .business_analytics import BusinessAnalytics
import time
import os
import threading
from queue import Queue
import torch
import torch.nn.modules.container as container
from collections import defaultdict

class VideoProcessor:
    def __init__(self, business_type='supermarket'):
        self.logger = logging.getLogger(__name__)
        
        # Carrega o modelo YOLO com verificação
        try:
            torch.serialization.add_safe_globals([DetectionModel, container.Sequential])
            self.model = YOLO('yolov8n.pt')
            if not hasattr(self.model, 'predict'):
                raise Exception("Modelo YOLO não carregado corretamente")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo YOLO: {str(e)}")
            raise
            
        self.business_analytics = BusinessAnalytics(business_type)
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_interval = 0.033  # ~30 FPS
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 100ms entre detecções
        self.frame_queue = Queue(maxsize=10)
        self.processing_thread = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.processing_delay = 0
        self.skip_frames = 0
        self.max_skip_frames = 3
        
        # Configurações de RTSP
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        # Configurações de classes e limiares por tipo de negócio
        self.class_configs = {
            'supermarket': {
                'classes': ['person', 'shopping cart', 'backpack', 'handbag', 'cell phone', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
                'thresholds': {
                    'person': 0.5,
                    'shopping cart': 0.4,
                    'backpack': 0.4,
                    'handbag': 0.4,
                    'cell phone': 0.4,
                    'bottle': 0.3,
                    'wine glass': 0.3,
                    'cup': 0.3,
                    'fork': 0.3,
                    'knife': 0.3,
                    'spoon': 0.3,
                    'bowl': 0.3
                }
            },
            'pharmacy': {
                'classes': ['person', 'backpack', 'handbag', 'chair', 'bottle', 'cell phone', 'book'],
                'thresholds': {
                    'person': 0.5,
                    'backpack': 0.4,
                    'handbag': 0.4,
                    'chair': 0.4,
                    'bottle': 0.3,
                    'cell phone': 0.4,
                    'book': 0.3
                }
            },
            'condominium': {
                'classes': ['person', 'car', 'truck', 'motorcycle', 'dog', 'cat', 'backpack', 'handbag', 'suitcase'],
                'thresholds': {
                    'person': 0.5,
                    'car': 0.4,
                    'truck': 0.4,
                    'motorcycle': 0.4,
                    'dog': 0.4,
                    'cat': 0.4,
                    'backpack': 0.4,
                    'handbag': 0.4,
                    'suitcase': 0.4
                }
            }
        }
        
        # Inicializa o rastreador de objetos
        self.tracked_objects = {}
        self.next_object_id = 0
        self.object_history = defaultdict(list)
        self.max_history_length = 30  # Mantém histórico dos últimos 30 frames
        
        # Configurações de cores para visualização
        self.colors = {
            'person': (0, 255, 0),      # Verde
            'car': (255, 0, 0),         # Azul
            'truck': (255, 0, 0),       # Azul
            'motorcycle': (255, 0, 0),  # Azul
            'dog': (0, 0, 255),         # Vermelho
            'cat': (0, 0, 255),         # Vermelho
            'backpack': (255, 255, 0),  # Ciano
            'handbag': (255, 255, 0),   # Ciano
            'suitcase': (255, 255, 0),  # Ciano
            'shopping cart': (0, 255, 255), # Amarelo
            'chair': (128, 0, 128),     # Roxo
            'cell phone': (255, 165, 0), # Laranja
            'default': (255, 255, 255)  # Branco
        }
        self.logger.info(f"Inicializado processador de vídeo para {business_type}")

    def connect(self, source):
        """Conecta à fonte de vídeo"""
        try:
            self.stop_detection()  # Para qualquer detecção em andamento
            
            # Verifica se é um arquivo local ou URL
            if os.path.isfile(source):
                self.cap = cv2.VideoCapture(source)
            else:
                # Configurações específicas para RTSP
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                raise Exception("Não foi possível abrir a fonte de vídeo")
            
            # Configurações de buffer e codec
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduzido para menor latência
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verifica se consegue ler o primeiro frame
            ret, frame = self.cap.read()
            if not ret or not self._is_valid_frame(frame):
                raise Exception("Não foi possível ler frames da fonte de vídeo")
            
            # Inicia o processamento
            self.is_running = True
            self.start_time = time.time()
            
            # Inicia thread de processamento
            self.processing_thread = threading.Thread(target=self._process_frames_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info(f"Conectado à fonte de vídeo: {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao conectar à fonte de vídeo: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def disconnect(self):
        """Desconecta da fonte de vídeo"""
        self.stop_detection()
        self.logger.info("Desconectado da fonte de vídeo")

    def stop_detection(self):
        """Para o processamento de vídeo"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None
        self.current_frame = None
        while not self.frame_queue.empty():
            self.frame_queue.get()
        self.logger.info("Detecção parada")

    def _is_valid_frame(self, frame):
        """Verifica se o frame é válido"""
        if frame is None or frame.size == 0:
            return False
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return False
        return True

    def _process_frames_thread(self):
        """Thread para processar frames em background"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    if self._is_valid_frame(frame):
                        # Calcula o atraso de processamento
                        current_time = time.time()
                        processing_time = current_time - self.last_frame_time
                        self.processing_delay = processing_time - self.frame_interval
                        
                        # Ajusta o número de frames a pular baseado no atraso
                        if self.processing_delay > 0.1:  # Se o atraso for maior que 100ms
                            self.skip_frames = min(self.max_skip_frames, int(self.processing_delay / self.frame_interval))
                        else:
                            self.skip_frames = 0
                        
                        # Processa o frame apenas se não estiver muito atrasado
                        if self.skip_frames == 0:
                            processed_frame = self.process_frame(frame)
                            if processed_frame is not None:
                                self.current_frame = processed_frame
                                self.last_frame_time = current_time
                        else:
                            self.logger.debug(f"Pulando {self.skip_frames} frames devido ao atraso de {self.processing_delay:.3f}s")
                            
                    self.frame_queue.task_done()
                time.sleep(0.001)  # Reduzido para melhor responsividade
            except Exception as e:
                self.logger.error(f"Erro no processamento de frames: {str(e)}")
                time.sleep(1)

    def _update_tracker(self, frame, detections):
        """Atualiza o rastreador de objetos"""
        current_objects = set()
        
        # Atualiza objetos existentes
        for obj_id, tracker in list(self.tracked_objects.items()):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                current_objects.add(obj_id)
                self.object_history[obj_id].append((x + w/2, y + h/2))
                if len(self.object_history[obj_id]) > self.max_history_length:
                    self.object_history[obj_id].pop(0)
            else:
                del self.tracked_objects[obj_id]
                del self.object_history[obj_id]
        
        # Adiciona novos objetos
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            
            # Verifica se o objeto já está sendo rastreado
            is_new = True
            for obj_id, tracker in self.tracked_objects.items():
                success, existing_bbox = tracker.update(frame)
                if success:
                    ex, ey, ew, eh = [int(v) for v in existing_bbox]
                    if abs(ex - x1) < 50 and abs(ey - y1) < 50:
                        is_new = False
                        break
            
            if is_new:
                # Cria um novo rastreador KCF
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                self.tracked_objects[self.next_object_id] = tracker
                self.object_history[self.next_object_id] = [(x1 + w/2, y1 + h/2)]
                self.next_object_id += 1

    def _draw_tracking(self, frame):
        """Desenha as trajetórias dos objetos rastreados"""
        for obj_id, history in self.object_history.items():
            if len(history) > 1:
                points = np.array(history, dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                
                # Desenha o ID do objeto
                if history:
                    x, y = history[-1]
                    cv2.putText(frame, f"ID: {obj_id}", (int(x), int(y)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def _determine_zone(self, x1, y1, x2, y2, frame_shape):
        """Determina a zona do objeto baseado em sua posição"""
        height, width = frame_shape[:2]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Divide o frame em zonas
        if center_x < width * 0.33:
            return 'entrance'
        elif center_x < width * 0.66:
            return 'middle'
        else:
            return 'exit'

    def process_frame(self, frame):
        """Processa um frame e retorna o frame com detecções"""
        try:
            if not self._is_valid_frame(frame):
                return frame
                
            current_time = time.time()
            
            # Limita a frequência de detecções
            if current_time - self.last_detection_time < self.detection_interval:
                return frame
                
            self.last_detection_time = current_time
            
            # Obtém configurações do tipo de negócio atual
            business_config = self.class_configs.get(self.business_analytics.business_type, self.class_configs['supermarket'])
            allowed_classes = business_config['classes']
            thresholds = business_config['thresholds']
            
            # Realiza a detecção
            results = self.model(frame, conf=0.3)  # Limiar inicial baixo para capturar mais detecções
            
            detections = []
            # Processa as detecções
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Extrai informações da detecção
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # Verifica se a classe é permitida e se atinge o limiar
                        if class_name in allowed_classes and confidence >= thresholds.get(class_name, 0.3):
                            # Verifica se as coordenadas estão dentro dos limites do frame
                            height, width = frame.shape[:2]
                            x1 = max(0, min(int(x1), width-1))
                            y1 = max(0, min(int(y1), height-1))
                            x2 = max(0, min(int(x2), width-1))
                            y2 = max(0, min(int(y2), height-1))
                            
                            # Determina a zona baseado na posição do objeto
                            zone = self._determine_zone(x1, y1, x2, y2, frame.shape)
                            
                            # Adiciona à lista de detecções
                            detections.append({
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'zone': zone
                            })
                            
                            # Atualiza métricas de negócio
                            self.business_analytics.process_detection(detections[-1])
                            
                            # Desenha a detecção no frame
                            color = self.colors.get(class_name, self.colors['default'])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                    except Exception as e:
                        self.logger.error(f"Erro ao processar detecção: {str(e)}")
                        continue
            
            # Atualiza o rastreador e desenha as trajetórias
            self._update_tracker(frame, detections)
            self._draw_tracking(frame)
                        
            return frame
            
        except Exception as e:
            self.logger.error(f"Erro ao processar frame: {str(e)}")
            return frame

    def generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            try:
                if not self.is_running or self.cap is None:
                    # Tela de espera quando não há vídeo
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Aguardando video...",
                              (200, 240), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255, 255, 255), 2)
                else:
                    ret, frame = self.cap.read()
                    if not ret or not self._is_valid_frame(frame):
                        self.logger.warning("Erro ao ler frame, tentando reconectar...")
                        time.sleep(1)
                        continue

                # Processa o frame
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    frame = processed_frame

                # Converte o frame para JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Pequeno delay para controlar o FPS
                time.sleep(self.frame_interval)

            except Exception as e:
                self.logger.error(f"Erro ao gerar frame: {str(e)}")
                time.sleep(1)

    def get_metrics(self):
        """Retorna as métricas atuais"""
        return self.business_analytics.get_business_insights() 