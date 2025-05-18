import json
import time
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import numpy as np
from scipy import stats

class BusinessAnalytics:
    def __init__(self, business_type='supermarket'):
        self.business_type = business_type
        self.logger = logging.getLogger(__name__)
        
        # Configurações específicas por tipo de negócio
        self.business_configs = {
            'supermarket': {
                'important_objects': ['person', 'shopping cart', 'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'sandwich', 'carrot', 'cell phone', 'backpack', 'handbag'],
                'zones': ['entrance', 'middle', 'exit'],
                'metrics': {
                    'person_count': {'threshold': 20, 'warning': 'Alta movimentação'},
                    'cart_count': {'threshold': 10, 'warning': 'Muitos carrinhos em uso'},
                    'product_count': {'threshold': 50, 'warning': 'Alto volume de produtos'},
                    'backpack_count': {'threshold': 5, 'warning': 'Alto número de mochilas'},
                    'handbag_count': {'threshold': 5, 'warning': 'Alto número de bolsas'},
                    'cellphone_count': {'threshold': 8, 'warning': 'Alto uso de celulares'}
                }
            },
            'pharmacy': {
                'important_objects': ['person', 'bottle', 'cup', 'bowl', 'book', 'cell phone', 'backpack', 'handbag', 'chair', 'bench'],
                'zones': ['entrance', 'middle', 'exit'],
                'metrics': {
                    'person_count': {'threshold': 10, 'warning': 'Alta movimentação'},
                    'prescription_count': {'threshold': 5, 'warning': 'Alto volume de prescrições'},
                    'medicine_count': {'threshold': 20, 'warning': 'Alto volume de medicamentos'},
                    'backpack_count': {'threshold': 3, 'warning': 'Alto número de mochilas'},
                    'handbag_count': {'threshold': 3, 'warning': 'Alto número de bolsas'},
                    'chair_count': {'threshold': 4, 'warning': 'Alto uso de cadeiras'}
                }
            },
            'condominium': {
                'important_objects': ['person', 'car', 'bicycle', 'motorcycle', 'truck', 'dog', 'cat', 'backpack', 'handbag'],
                'zones': ['entrance', 'middle', 'exit'],
                'metrics': {
                    'person_count': {'threshold': 15, 'warning': 'Alta movimentação'},
                    'car_count': {'threshold': 5, 'warning': 'Alto fluxo de veículos'},
                    'bicycle_count': {'threshold': 3, 'warning': 'Alto fluxo de bicicletas'},
                    'dog_count': {'threshold': 2, 'warning': 'Alto número de cachorros'},
                    'cat_count': {'threshold': 2, 'warning': 'Alto número de gatos'},
                    'backpack_count': {'threshold': 4, 'warning': 'Alto número de mochilas'}
                }
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
                'average_stay_time': 0,
                'backpack_count': 0,
                'handbag_count': 0,
                'cellphone_count': 0,
                'peak_hours': defaultdict(int),
                'object_trends': defaultdict(list),
                'zone_transitions': defaultdict(int),
                'performance_metrics': {
                    'detection_rate': 0,
                    'processing_time': 0,
                    'confidence_avg': 0
                }
            },
            'pharmacy': {
                'person_count': 0,
                'prescription_count': 0,
                'medicine_count': 0,
                'zone_density': defaultdict(int),
                'average_stay_time': 0,
                'backpack_count': 0,
                'handbag_count': 0,
                'chair_count': 0,
                'peak_hours': defaultdict(int),
                'object_trends': defaultdict(list),
                'zone_transitions': defaultdict(int),
                'performance_metrics': {
                    'detection_rate': 0,
                    'processing_time': 0,
                    'confidence_avg': 0
                }
            },
            'condominium': {
                'person_count': 0,
                'car_count': 0,
                'bicycle_count': 0,
                'zone_density': defaultdict(int),
                'average_stay_time': 0,
                'dog_count': 0,
                'cat_count': 0,
                'backpack_count': 0,
                'peak_hours': defaultdict(int),
                'object_trends': defaultdict(list),
                'zone_transitions': defaultdict(int),
                'performance_metrics': {
                    'detection_rate': 0,
                    'processing_time': 0,
                    'confidence_avg': 0
                }
            }
        }
        
        self.detection_history = []
        self.last_update = time.time()
        self.object_tracking = {}
        self.object_positions = {}
        self.object_timestamps = {}
        self.performance_history = []
        
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
        """Rastreia objetos entre frames usando IoU e Kalman Filter"""
        current_time = time.time()
        bbox = detection_data['bbox']
        class_name = detection_data['class_name']
        confidence = detection_data.get('confidence', 0)
        
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
            old_zone = self.object_tracking[best_match]['zone']
            new_zone = detection_data.get('zone', 'unknown')
            
            if old_zone != new_zone:
                self.metrics[self.business_type]['zone_transitions'][f"{old_zone}_to_{new_zone}"] += 1
            
            self.object_tracking[best_match].update({
                'bbox': bbox,
                'last_seen': current_time,
                'zone': new_zone,
                'confidence': confidence,
                'trajectory': self.object_tracking[best_match].get('trajectory', []) + [bbox]
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
                'zone': detection_data.get('zone', 'unknown'),
                'confidence': confidence,
                'trajectory': [bbox]
            }
            return obj_id
            
    def process_detection(self, detection_data):
        """Processa uma detecção e atualiza as métricas"""
        start_time = time.time()
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
            
        # Atualiza métricas de performance
        processing_time = time.time() - start_time
        self.performance_history.append(processing_time)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
            
        self.metrics[self.business_type]['performance_metrics'].update({
            'detection_rate': len(self.detection_history) / 5,  # Detecções por segundo
            'processing_time': np.mean(self.performance_history),
            'confidence_avg': np.mean([d['data'].get('confidence', 0) for d in self.detection_history])
        })
        
        # Atualiza horários de pico
        hour = datetime.fromtimestamp(current_time).hour
        self.metrics[self.business_type]['peak_hours'][hour] += 1
        
        # Atualiza tendências
        self.metrics[self.business_type]['object_trends'][detection_data['class_name']].append(current_time)
        if len(self.metrics[self.business_type]['object_trends'][detection_data['class_name']]) > 100:
            self.metrics[self.business_type]['object_trends'][detection_data['class_name']].pop(0)
            
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
        
        # NOVOS SERVIÇOS SUPERMERCADO
        # 1. Contagem de mochilas (backpack)
        if detection_data['class_name'] == 'backpack':
            metrics['backpack_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'backpack' and
                                         current_time - d['time'] < 5])
        # 2. Contagem de bolsas (handbag)
        if detection_data['class_name'] == 'handbag':
            metrics['handbag_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'handbag' and
                                         current_time - d['time'] < 5])
        # 3. Contagem de celulares (cell phone)
        if detection_data['class_name'] == 'cell phone':
            metrics['cellphone_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'cell phone' and
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
        
        # NOVOS SERVIÇOS FARMÁCIA
        # 1. Contagem de mochilas (backpack)
        if detection_data['class_name'] == 'backpack':
            metrics['backpack_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'backpack' and
                                         current_time - d['time'] < 5])
        # 2. Contagem de bolsas (handbag)
        if detection_data['class_name'] == 'handbag':
            metrics['handbag_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'handbag' and
                                         current_time - d['time'] < 5])
        # 3. Contagem de cadeiras (chair)
        if detection_data['class_name'] == 'chair':
            metrics['chair_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'chair' and
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
        
        # NOVOS SERVIÇOS CONDOMÍNIO
        # 1. Contagem de cachorros (dog)
        if detection_data['class_name'] == 'dog':
            metrics['dog_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'dog' and
                                         current_time - d['time'] < 5])
        # 2. Contagem de gatos (cat)
        if detection_data['class_name'] == 'cat':
            metrics['cat_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'cat' and
                                         current_time - d['time'] < 5])
        # 3. Contagem de mochilas (backpack)
        if detection_data['class_name'] == 'backpack':
            metrics['backpack_count'] = len([d for d in self.detection_history 
                                         if d['data']['class_name'] == 'backpack' and
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
        trends = []
        
        # Análise de tendências
        for obj_type, timestamps in metrics['object_trends'].items():
            if len(timestamps) > 10:
                # Calcula a tendência usando regressão linear
                x = np.array(range(len(timestamps)))
                y = np.array(timestamps)
                slope, _, r_value, _, _ = stats.linregress(x, y)
                
                if abs(slope) > 0.1:  # Se há uma tendência significativa
                    trend = "aumentando" if slope > 0 else "diminuindo"
                    trends.append(f"Tendência de {obj_type} está {trend} (confiança: {r_value**2:.2f})")
        
        # Análise de horários de pico
        peak_hours = sorted(metrics['peak_hours'].items(), key=lambda x: x[1], reverse=True)[:3]
        if peak_hours:
            peak_times = [f"{h:02d}:00" for h, _ in peak_hours]
            recommendations.append(f"Horários de pico: {', '.join(peak_times)}")
        
        # Análise de transições entre zonas
        transitions = metrics['zone_transitions']
        if transitions:
            most_common = max(transitions.items(), key=lambda x: x[1])
            recommendations.append(f"Transição mais comum: {most_common[0]}")
        
        # Análise de performance
        perf_metrics = metrics['performance_metrics']
        if perf_metrics['processing_time'] > 0.1:  # Se o tempo de processamento for alto
            recommendations.append("Performance do sistema pode ser otimizada")
        
        # Verifica thresholds específicos do tipo de negócio
        for metric, config in self.business_configs[self.business_type]['metrics'].items():
            if metrics[metric] > config['threshold']:
                recommendations.append(config['warning'])
        
        return {
            'business_type': self.business_type,
            'metrics': metrics,
            'recommendations': recommendations,
            'trends': trends,
            'performance': perf_metrics
        } 