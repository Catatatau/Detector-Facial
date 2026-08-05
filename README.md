<div align="center">

# Enterprise Local Face Recognition Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-red.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0+-green.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/PySide6-GUI-orange.svg?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
</p>

Uma plataforma corporativa, modular e 100% offline para reconhecimento facial. Construida utilizando tecnologias modernas de visao computacional, bancos de dados robustos e uma interface grafica amigavel.

[Caracteristicas](#caracteristicas) - 
[Como Instalar](#instalacao) - 
[Uso](#como-usar) - 
[Arquitetura](#arquitetura)

---
</div>

## Caracteristicas

* **Local-First (100% Offline):** Sem dependencias de nuvem. Toda a inferencia e armazenamento de dados ocorre localmente, garantindo total privacidade e conformidade com LGPD.
* **Arquitetura Modular:** Facilidade extrema para alternar entre diferentes detectores de face (YuNet, OpenCV Haar) e modelos de extracao de caracteristicas biometricas (InsightFace).
* **Alta Performance:** Captura de camera otimizada com multiplas threads, rastreamento inteligente e inferencia rapida.
* **Interface Corporativa:** Design moderno, limpo e profissional construido utilizando a biblioteca Qt via PySide6.
* **Banco de Dados Robusto:** Gerenciamento seguro de dados via ORM SQLAlchemy e suporte a migracoes com Alembic (Utiliza SQLite como padrao, escalavel para PostgreSQL/MySQL).

<br>

## Instalacao

### Pre-requisitos
* [Python 3.12+](https://www.python.org/downloads/)
* Uma webcam conectada e configurada

### Passos para Instalacao

1. **Crie e ative um ambiente virtual (Recomendado):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Instale as dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inicialize e migre o banco de dados:**
   ```bash
   python -m alembic upgrade head
   ```

<br>

## Como Usar

Para iniciar a interface grafica da aplicacao, basta rodar o script principal:

```bash
python main.py
```

### Configuracao do Sistema

A configuracao e gerada automaticamente no primeiro uso dentro do arquivo `config.json` (ou pode ser configurada via `.env`). La, voce podera ajustar parametros cruciais como:
- **Camera:** Indice, resolucao e FPS.
- **Modelos:** Detector ativo e limites de reconhecimento (thresholds).
- **Sistema:** Niveis de log e processamento em threads.

<br>

## Arquitetura

O projeto foi organizado buscando a melhor manutenibilidade e separacao de responsabilidades:

```text
face_platform/
|-- camera/         # Captura de video multithread
|-- detectors/      # Abstracao de algoritmos de deteccao (YuNet, etc)
|-- embeddings/     # Modelos de caracteristicas faciais (InsightFace)
|-- database/       # Modelos do SQLAlchemy (Users, Embeddings, Events)
|-- ui/             # Componentes de interface com PySide6
|-- config/         # Gerenciamento de preferencias do usuario
```

<br>

---
<div align="center">
  Desenvolvido com dedicacao para seguranca biometrica local.
</div>
