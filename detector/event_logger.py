import os
import time
import json
from datetime import datetime
import logging
from collections import deque

class EventLogger:
    def __init__(self, log_file='events/events.log'):
        """Initialize the event logger with a log file path"""
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.max_events = 1000  # Número máximo de eventos em memória
        self.events_buffer = deque(maxlen=self.max_events)
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Criar arquivo se não existir
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')
        
        # Carregar eventos existentes
        self._load_existing_events()
        
        self.logger.info(f"EventLogger inicializado com arquivo: {log_file}")

    def _parse_log_line(self, line):
        """Parse a log line in either JSON or CSV format"""
        line = line.strip()
        if not line:
            return None
            
        try:
            # Try JSON format first
            return json.loads(line)
        except json.JSONDecodeError:
            try:
                # Try CSV format (timestamp,object_type,confidence)
                parts = line.split(',')
                if len(parts) >= 3:
                    timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S').timestamp()
                    return {
                        'timestamp': timestamp,
                        'datetime': parts[0],
                        'type': 'detection',
                        'data': {
                            'class_name': parts[1],
                            'confidence': float(parts[2])
                        }
                    }
            except (ValueError, IndexError):
                pass
                
        return None

    def _load_existing_events(self):
        """Carrega eventos existentes do arquivo de log"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    for line in f:
                        event = self._parse_log_line(line)
                        if event:
                            self.events_buffer.append(event)
                        else:
                            self.logger.warning(f"Linha inválida no arquivo de log: {line}")
        except Exception as e:
            self.logger.error(f"Erro ao carregar eventos existentes: {str(e)}")

    def log_event(self, event_type, data, confidence=None, timestamp=None):
        """Registra um evento de detecção com dados adicionais"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Criar evento
            event = {
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'type': event_type,
                'data': data
            }
            
            if confidence is not None:
                event['confidence'] = confidence
            
            # Adicionar ao buffer em memória
            self.events_buffer.append(event)
            
            # Escrever no arquivo em formato JSON
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
            
            self.logger.debug(f"Evento registrado: {event}")
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar evento: {str(e)}")

    def get_events(self, limit=100, event_type=None, start_time=None, end_time=None):
        """Obtém eventos filtrados do buffer em memória"""
        try:
            events = list(self.events_buffer)
            
            # Aplicar filtros
            if event_type:
                events = [e for e in events if e['type'] == event_type]
            
            if start_time:
                events = [e for e in events if e['timestamp'] >= start_time]
            
            if end_time:
                events = [e for e in events if e['timestamp'] <= end_time]
            
            # Ordenar por timestamp (mais recentes primeiro)
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limitar número de eventos
            return events[:limit]
            
        except Exception as e:
            self.logger.error(f"Erro ao obter eventos: {str(e)}")
            return []

    def get_event_stats(self, event_type=None, time_window=3600):
        """Obtém estatísticas dos eventos"""
        try:
            current_time = time.time()
            start_time = current_time - time_window
            
            events = self.get_events(
                limit=None,
                event_type=event_type,
                start_time=start_time,
                end_time=current_time
            )
            
            stats = {
                'total_events': len(events),
                'events_per_type': {},
                'average_confidence': 0,
                'time_window': time_window
            }
            
            if events:
                # Contagem por tipo
                for event in events:
                    event_type = event['type']
                    stats['events_per_type'][event_type] = stats['events_per_type'].get(event_type, 0) + 1
                
                # Média de confiança
                confidences = []
                for event in events:
                    if 'confidence' in event:
                        confidences.append(event['confidence'])
                    elif 'data' in event and 'confidence' in event['data']:
                        confidences.append(event['data']['confidence'])
                        
                if confidences:
                    stats['average_confidence'] = sum(confidences) / len(confidences)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return {
                'total_events': 0,
                'events_per_type': {},
                'average_confidence': 0,
                'time_window': time_window
            }

    def clear_events(self):
        """Limpa todos os eventos do arquivo de log e do buffer"""
        try:
            with open(self.log_file, 'w') as f:
                f.write('')
            self.events_buffer.clear()
            self.logger.info("Eventos limpos com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao limpar eventos: {str(e)}")

    def get_recent_events(self, limit=100):
        """Alias para get_events para manter compatibilidade"""
        return self.get_events(limit)

    def export_events(self, filepath, format='json'):
        """Exporta eventos para um arquivo"""
        try:
            events = list(self.events_buffer)
            
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(events, f, indent=2)
            elif format == 'csv':
                with open(filepath, 'w') as f:
                    f.write('timestamp,datetime,type,confidence,data\n')
                    for event in events:
                        confidence = event.get('confidence', '')
                        if not confidence and 'data' in event and 'confidence' in event['data']:
                            confidence = event['data']['confidence']
                        data = json.dumps(event.get('data', {}))
                        f.write(f"{event['timestamp']},{event['datetime']},{event['type']},{confidence},{data}\n")
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            self.logger.info(f"Eventos exportados com sucesso para: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar eventos: {str(e)}")
            return False 