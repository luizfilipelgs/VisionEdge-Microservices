import json
import time
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import numpy as np

class BusinessAnalytics:
    def __init__(self, business_type='supermarket'):
        self.business_type = business_type
        self.logger = logging.getLogger(__name__)
        
        # Configurações específicas por tipo de negócio
        self.business_configs = {
            'supermarket': {
                'important_objects': ['person', 'shopping cart', 'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'sandwich', 'carrot'],
                'zones': ['entrance', 'middle', 'exit']
            },
            'pharmacy': {
                'important_objects': ['person', 'bottle', 'cup', 'bowl', 'book', 'cell phone'],
                'zones': ['entrance', 'middle', 'exit']
            },
            'condominium': {
                'important_objects': ['person', 'car', 'bicycle', 'motorcycle', 'truck'],
                'zones': ['entrance', 'middle', 'exit']
            }
        }
        
        # Inicializa contadores e métricas
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reseta todas as métricas para o estado inicial"""
        self.metrics = {
            'supermarket': {
                'person_count': 0,
                'cart_count': 0,
                'product_count': 0,
                'zone_density': defaultdict(int),
                'average_stay_time': 0
            },
            'pharmacy': {
                'person_count': 0,
                'prescription_count': 0,
                'medicine_count': 0,
                'zone_density': defaultdict(int),
                'average_stay_time': 0
            },
            'condominium': {
                'person_count': 0,
                'car_count': 0,
                'bicycle_count': 0,
                'zone_density': defaultdict(int),
                'average_stay_time': 0
            }
        }
        
        self.detection_history = []
        self.last_update = time.time()
        self.object_tracking = {}  # Para rastrear objetos entre frames
        self.object_positions = {}  # Para rastrear posições dos objetos
        self.object_timestamps = {}  # Para rastrear timestamps dos objetos
        
    def _calculate_iou(self, box1, box2):
        """Calcula o IoU (Intersection over Union) entre duas bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def _track_object(self, detection_data):
        """Rastreia objetos entre frames usando IoU"""
        current_time = time.time()
        bbox = detection_data['bbox']
        class_name = detection_data['class_name']
        
        # Limpa objetos antigos
        self.object_tracking = {k: v for k, v in self.object_tracking.items() 
                              if current_time - v['last_seen'] < 5}
        
        # Procura por correspondências
        best_match = None
        best_iou = 0.5  # Threshold mínimo de IoU
        
        for obj_id, obj_data in self.object_tracking.items():
            if obj_data['class'] == class_name:
                iou = self._calculate_iou(bbox, obj_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = obj_id
        
        if best_match is not None:
            # Atualiza objeto existente
            self.object_tracking[best_match].update({
                'bbox': bbox,
                'last_seen': current_time,
                'zone': detection_data.get('zone', 'unknown')
            })
            return best_match
        else:
            # Cria novo objeto
            obj_id = f"{class_name}_{len(self.object_tracking)}"
            self.object_tracking[obj_id] = {
                'class': class_name,
                'bbox': bbox,
                'first_seen': current_time,
                'last_seen': current_time,
                'zone': detection_data.get('zone', 'unknown')
            }
            return obj_id
            
    def process_detection(self, detection_data):
        """Processa uma detecção e atualiza as métricas"""
        current_time = time.time()
        
        # Rastreia o objeto
        obj_id = self._track_object(detection_data)
        
        # Atualiza histórico de detecções
        self.detection_history.append({
            'time': current_time,
            'data': detection_data,
            'obj_id': obj_id
        })
        
        # Remove detecções antigas (mais de 5 segundos)
        self.detection_history = [d for d in self.detection_history 
                                if current_time - d['time'] < 5]
        
        # Atualiza métricas específicas do tipo de negócio
        if self.business_type == 'supermarket':
            self._update_supermarket_metrics(detection_data, obj_id)
        elif self.business_type == 'pharmacy':
            self._update_pharmacy_metrics(detection_data, obj_id)
        elif self.business_type == 'condominium':
            self._update_condominium_metrics(detection_data, obj_id)
            
        self.last_update = current_time
        
    def _update_supermarket_metrics(self, detection_data, obj_id):
        """Atualiza métricas específicas para supermercado"""
        metrics = self.metrics['supermarket']
        current_time = time.time()
        
        # Contagem de pessoas
        if detection_data['class_name'] == 'person':
            metrics['person_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'person' and
                                         current_time - d['time'] < 5])
        
        # Contagem de carrinhos
        if detection_data['class_name'] == 'shopping cart':
            metrics['cart_count'] = len([d for d in self.detection_history 
                                       if d['data']['class_name'] == 'shopping cart' and
                                       current_time - d['time'] < 5])
        
        # Contagem de produtos
        if detection_data['class_name'] in ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'sandwich', 'carrot']:
            metrics['product_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] in ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'sandwich', 'carrot'] and
                                         current_time - d['time'] < 5])
        
        # Densidade por zona
        if 'zone' in detection_data:
            metrics['zone_density'][detection_data['zone']] = len([d for d in self.detection_history 
                                                                 if d['data'].get('zone') == detection_data['zone'] and
                                                                 current_time - d['time'] < 5])
        
        # Tempo médio de permanência
        stay_times = []
        for obj_id, obj_data in self.object_tracking.items():
            if obj_data['class'] == 'person':
                stay_time = obj_data['last_seen'] - obj_data['first_seen']
                if stay_time < 300:  # Ignora tempos muito longos
                    stay_times.append(stay_time)
        
        if stay_times:
            metrics['average_stay_time'] = np.mean(stay_times)
        
    def _update_pharmacy_metrics(self, detection_data, obj_id):
        """Atualiza métricas específicas para farmácia"""
        metrics = self.metrics['pharmacy']
        current_time = time.time()
        
        # Contagem de pessoas
        if detection_data['class_name'] == 'person':
            metrics['person_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'person' and
                                         current_time - d['time'] < 5])
        
        # Contagem de prescrições (simulado)
        if detection_data['class_name'] == 'book':
            metrics['prescription_count'] = len([d for d in self.detection_history 
                                              if d['data']['class_name'] == 'book' and
                                              current_time - d['time'] < 5])
        
        # Contagem de medicamentos
        if detection_data['class_name'] in ['bottle', 'cup', 'bowl']:
            metrics['medicine_count'] = len([d for d in self.detection_history 
                                          if d['data']['class_name'] in ['bottle', 'cup', 'bowl'] and
                                          current_time - d['time'] < 5])
        
        # Densidade por zona
        if 'zone' in detection_data:
            metrics['zone_density'][detection_data['zone']] = len([d for d in self.detection_history 
                                                                 if d['data'].get('zone') == detection_data['zone'] and
                                                                 current_time - d['time'] < 5])
        
        # Tempo médio de permanência
        stay_times = []
        for obj_id, obj_data in self.object_tracking.items():
            if obj_data['class'] == 'person':
                stay_time = obj_data['last_seen'] - obj_data['first_seen']
                if stay_time < 300:  # Ignora tempos muito longos
                    stay_times.append(stay_time)
        
        if stay_times:
            metrics['average_stay_time'] = np.mean(stay_times)
        
    def _update_condominium_metrics(self, detection_data, obj_id):
        """Atualiza métricas específicas para condomínio"""
        metrics = self.metrics['condominium']
        current_time = time.time()
        
        # Contagem de pessoas
        if detection_data['class_name'] == 'person':
            metrics['person_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'person' and
                                         current_time - d['time'] < 5])
        
        # Contagem de carros
        if detection_data['class_name'] == 'car':
            metrics['car_count'] = len([d for d in self.detection_history 
                                      if d['data']['class_name'] == 'car' and
                                      current_time - d['time'] < 5])
        
        # Contagem de bicicletas
        if detection_data['class_name'] == 'bicycle':
            metrics['bicycle_count'] = len([d for d in self.detection_history 
                                          if d['data']['class_name'] == 'bicycle' and
                                          current_time - d['time'] < 5])
        
        # Densidade por zona
        if 'zone' in detection_data:
            metrics['zone_density'][detection_data['zone']] = len([d for d in self.detection_history 
                                                                 if d['data'].get('zone') == detection_data['zone'] and
                                                                 current_time - d['time'] < 5])
        
        # Tempo médio de permanência
        stay_times = []
        for obj_id, obj_data in self.object_tracking.items():
            if obj_data['class'] == 'person':
                stay_time = obj_data['last_seen'] - obj_data['first_seen']
                if stay_time < 300:  # Ignora tempos muito longos
                    stay_times.append(stay_time)
        
        if stay_times:
            metrics['average_stay_time'] = np.mean(stay_times)
        
    def get_business_insights(self):
        """Retorna insights de negócio baseados nas métricas atuais"""
        metrics = self.metrics[self.business_type]
        recommendations = []
        
        if self.business_type == 'supermarket':
            if metrics['person_count'] > 20:
                recommendations.append("Alta movimentação detectada. Considere abrir mais caixas.")
            
            if metrics['cart_count'] > 10:
                recommendations.append("Muitos carrinhos em uso. Verifique a disponibilidade.")
            
            if metrics['product_count'] > 50:
                recommendations.append("Alto volume de produtos detectados. Verifique o estoque.")
            
            if metrics['zone_density']['entrance'] > 5:
                recommendations.append("Congestionamento na entrada. Considere abrir mais portas.")
            
            if metrics['average_stay_time'] > 600:  # 10 minutos
                recommendations.append("Tempo médio de permanência alto. Otimize o layout da loja.")
            
        elif self.business_type == 'pharmacy':
            if metrics['person_count'] > 10:
                recommendations.append("Alta movimentação na farmácia. Considere aumentar a equipe.")
            
            if metrics['prescription_count'] > 5:
                recommendations.append("Alto volume de prescrições. Otimize o atendimento.")
            
            if metrics['medicine_count'] > 20:
                recommendations.append("Alto volume de medicamentos. Verifique o estoque.")
            
            if metrics['average_stay_time'] > 300:  # 5 minutos
                recommendations.append("Tempo de permanência alto. Otimize o atendimento.")
            
        elif self.business_type == 'condominium':
            if metrics['person_count'] > 15:
                recommendations.append("Alta movimentação de pessoas. Reforce a segurança.")
            
            if metrics['car_count'] > 5:
                recommendations.append("Alto fluxo de veículos. Verifique a capacidade do estacionamento.")
            
            if metrics['bicycle_count'] > 3:
                recommendations.append("Alto fluxo de bicicletas. Verifique a segurança cicloviária.")
            
            if metrics['zone_density']['entrance'] > 5:
                recommendations.append("Congestionamento na entrada. Considere abrir mais portas.")
            
            if metrics['average_stay_time'] > 1800:  # 30 minutos
                recommendations.append("Tempo médio de permanência alto. Verifique a segurança.")
        
        return {
            'business_type': self.business_type,
            'metrics': metrics,
            'recommendations': recommendations
        } 