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
        self.frame_queue = Queue(maxsize=10)  # Aumentado para 10 frames
        self.processing_thread = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.processing_delay = 0
        self.skip_frames = 0
        self.max_skip_frames = 3
        self.logger.info(f"Inicializado processador de vídeo para {business_type}")
        
        # Configurações de RTSP
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

    def _connect_to_stream(self, url):
        """Tenta conectar ao stream com retentativas"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Tentativa {attempt + 1} de conectar ao stream: {url}")
                
                # Configurações específicas para RTSP
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                
                # Configurações de buffer e codec otimizadas
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Aumentado para 5
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolução padrão
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                if not cap.isOpened():
                    raise Exception("Não foi possível abrir o stream")
                
                # Tenta ler um frame com timeout
                start_time = time.time()
                while time.time() - start_time < 5:  # 5 segundos de timeout
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.logger.info("Conexão com o stream estabelecida com sucesso")
                        return cap
                    time.sleep(0.1)
                
                raise Exception("Timeout ao tentar ler frame inicial")
                
            except Exception as e:
                self.logger.error(f"Erro na tentativa {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"Falha ao conectar ao stream após {max_retries} tentativas: {str(e)}")

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

    def start_stream_detection(self, stream_url):
        """Start detection on a video stream"""
        try:
            self.stop_detection()
            self.cap = self._connect_to_stream(stream_url)
            self.is_running = True
            
            # Inicia thread de processamento
            self.processing_thread = threading.Thread(target=self._process_frames_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info(f"Iniciando detecção no stream: {stream_url}")
        except Exception as e:
            self.logger.error(f"Erro ao iniciar detecção em stream: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            raise

    def start_file_detection(self, filepath):
        """Start detection on a video file"""
        try:
            self.stop_detection()
            self.cap = cv2.VideoCapture(filepath)
            if not self.cap.isOpened():
                raise Exception(f"Não foi possível abrir o vídeo: {filepath}")
            self.is_running = True
            
            # Inicia thread de processamento
            self.processing_thread = threading.Thread(target=self._process_frames_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info(f"Iniciando detecção no arquivo: {filepath}")
        except Exception as e:
            self.logger.error(f"Erro ao iniciar detecção em arquivo: {str(e)}")
            raise

    def stop_detection(self):
        """Stop the current detection process"""
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

    def process_frame(self, frame):
        """Processa um frame e retorna o frame com detecções"""
        try:
            if not self._is_valid_frame(frame):
                return frame
                
            current_time = time.time()
            
            # Limita a frequência de detecções para melhor performance
            if current_time - self.last_detection_time < self.detection_interval:
                return frame
                
            self.last_detection_time = current_time
            
            # Realiza a detecção com confiança mínima de 0.5 (aumentado de 0.4)
            results = self.model(frame, conf=0.5)
            
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
                        
                        # Verifica se as coordenadas estão dentro dos limites do frame
                        height, width = frame.shape[:2]
                        x1 = max(0, min(int(x1), width-1))
                        y1 = max(0, min(int(y1), height-1))
                        x2 = max(0, min(int(x2), width-1))
                        y2 = max(0, min(int(y2), height-1))
                        
                        # Determina a zona baseado na posição do objeto
                        zone = self._determine_zone(x1, y1, x2, y2, frame.shape)
                        
                        # Atualiza métricas de negócio
                        detection_data = {
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'zone': zone
                        }
                        self.business_analytics.process_detection(detection_data)
                        
                        # Desenha a detecção no frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    except Exception as e:
                        self.logger.error(f"Erro ao processar detecção: {str(e)}")
                        continue
                        
            return frame
            
        except Exception as e:
            self.logger.error(f"Erro ao processar frame: {str(e)}")
            return frame

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

    def generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            try:
                if self.is_running and self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret and self._is_valid_frame(frame):
                        # Limpa a fila se estiver cheia
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                pass
                                
                        # Adiciona frame à fila
                        self.frame_queue.put(frame)
                        
                        # Usa o frame processado mais recente
                        if self.current_frame is not None:
                            frame = self.current_frame
                            
                        # Atualiza contadores
                        self.frame_count += 1
                        current_time = time.time()
                        if current_time - self.last_frame_time >= 1.0:
                            fps = self.frame_count / (current_time - self.last_frame_time)
                            self.logger.debug(f"FPS atual: {fps:.2f}")
                            self.frame_count = 0
                            self.last_frame_time = current_time
                    else:
                        # Se não conseguir ler o frame, tenta reconectar
                        self.logger.warning("Erro ao ler frame, tentando reconectar...")
                        time.sleep(1)
                        continue
                else:
                    # Se não estiver rodando, mostra tela de espera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Aguardando video...",
                              (200, 240), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255, 255, 255), 2)

                # Converte o frame para JPEG com qualidade reduzida
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
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

    def get_business_insights(self):
        """Retorna os insights de negócio atuais"""
        return self.business_analytics.get_business_insights() 