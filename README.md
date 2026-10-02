# Detecção de EPIs com rastreamento

Projeto de visão computacional em que detecto capacete e colete em trabalhadores, atribuo um ID a cada pessoa com ByteTrack e registro quem ficou sem capacete e por quanto tempo.

Em desenvolvimento. Resultados, instruções de reprodução e limitações serão documentados conforme as etapas forem concluídas.

## Ambiente

Python 3.11, PyTorch 2.14.1 com CUDA 12.6 e Ultralytics 8.4.171, rodando em uma RTX 3060 Ti (8 GB) no Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_gpu.py
```

## Licença

AGPL-3.0, porque o projeto usa o Ultralytics YOLO, distribuído sob essa licença.
