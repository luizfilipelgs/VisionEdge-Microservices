# VisionEdge - Sistema de Visão Computacional

Sistema avançado de análise de vídeo em tempo real com detecção de objetos e métricas de negócio, desenvolvido para otimizar operações em diferentes tipos de estabelecimentos.

## 🌟 Funcionalidades Principais

### 🎯 Detecção em Tempo Real
- Detecção de objetos usando YOLOv8
- Suporte a múltiplos tipos de fontes de vídeo:
  - Streams RTSP
  - Câmeras IP
  - Arquivos de vídeo
  - Webcams

### 📊 Análise por Tipo de Negócio

#### 🏪 Supermercado
- Contagem de pessoas
- Monitoramento de carrinhos
- Taxa de ocupação
- Tempo médio de permanência
- Análise de fluxo de clientes

#### 💊 Farmácia
- Contagem de pessoas
- Tempo de espera
- Tamanho da fila
- Número de atendentes
- Análise de eficiência

#### 🏢 Condomínio
- Controle de acesso
- Monitoramento de veículos
- Contagem de entradas/saídas
- Alertas de segurança
- Análise de tráfego

### 💻 Interface Web
- Design responsivo e moderno
- Visualização em tempo real
- Métricas dinâmicas
- Recomendações inteligentes
- Exportação de relatórios

## 🚀 Requisitos do Sistema

### Hardware
- CPU: Intel i5/Ryzen 5 ou superior
- RAM: 8GB mínimo (16GB recomendado)
- GPU: NVIDIA com CUDA (opcional, mas recomendado)
- Armazenamento: 10GB livres

### Software
- Python 3.8 ou superior
- CUDA Toolkit 11.0+ (para GPU)
- Sistema operacional: Windows 10/11, Linux ou macOS
- Navegador web moderno (Chrome, Firefox, Edge)

## 📥 Instalação

1. **Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/visionedge.git
cd visionedge
```

2. **Configure o Ambiente Virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as Dependências**
```bash
pip install -r requirements.txt
```

4. **Baixe o Modelo YOLOv8**
```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "yolov8n.pt"

# Linux/Mac
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 🎮 Como Usar

1. **Inicie o Servidor**
```bash
python app.py
```

2. **Acesse a Interface**
```
http://localhost:5000
```

3. **Configure a Fonte de Vídeo**
- **Webcam**: Use `0` como URL
- **Câmera IP**: URL RTSP (ex: `rtsp://admin:admin@192.168.1.100:554/stream`)
- **Arquivo**: Faça upload do arquivo de vídeo

4. **Selecione o Tipo de Negócio**
- Escolha entre Supermercado, Farmácia ou Condomínio
- As métricas serão atualizadas automaticamente

5. **Inicie a Detecção**
- Clique em "Iniciar Detecção"
- Monitore as métricas em tempo real
- Acompanhe as recomendações

## 📁 Estrutura do Projeto

```
visionedge/
├── app.py                    # Aplicação Flask principal
├── requirements.txt          # Dependências do projeto
├── yolov8n.pt               # Modelo YOLOv8
├── detector/                # Módulo de detecção
│   ├── __init__.py
│   ├── video_processor.py   # Processamento de vídeo
│   ├── business_analytics.py # Análise de negócio
│   └── event_logger.py      # Registro de eventos
├── static/                  # Arquivos estáticos
├── templates/               # Templates HTML
│   └── index.html          # Interface principal
├── uploads/                # Pasta para uploads
└── events/                 # Logs de eventos
```

## ⚙️ Configuração

### Variáveis de Ambiente
```env
FLASK_ENV=development
FLASK_DEBUG=1
MODEL_PATH=yolov8n.pt
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
```

### Configurações Avançadas
- Ajuste de sensibilidade de detecção
- Configuração de zonas de interesse
- Personalização de alertas
- Exportação de dados

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

- **Issues**: [GitHub Issues](https://github.com/seu-usuario/visionedge/issues)
- **Email**: suporte@visionedge.com
- **Documentação**: [Wiki](https://github.com/seu-usuario/visionedge/wiki)

## 🙏 Agradecimentos

- Equipe Ultralytics pelo YOLOv8
- Comunidade Python
- Todos os contribuidores

---

Desenvolvido com ❤️ pela equipe VisionEdge 