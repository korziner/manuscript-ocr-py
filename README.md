# manuscript-ocr-py
Минималистичная CLI-распознавалка моделями konstantinkozhin/manuscript-ocr

<img width="1268" height="628" alt="image" src="https://github.com/user-attachments/assets/e111f09d-6e05-4685-9b0e-e7c0bfc0af45" />

```
usage: claude-opus-4-6.адаптивbatchTRBA.py [-h] [--device {cuda,cpu}] [--cache-dir CACHE_DIR] [--log-dir LOG_DIR] [--selftrain-dir SELFTRAIN_DIR]
                                           [--init-batch INIT_BATCH] [--min-batch MIN_BATCH] [--gpu-index GPU_INDEX] [--conf-high CONF_HIGH]
                                           [--conf-mid CONF_MID]
                                           input

positional arguments:
  input                 Файл или директория с изображениями

optional arguments:
  -h, --help            show this help message and exit
  --device {cuda,cpu}   Основное устройство для моделей (cuda/cpu)
  --cache-dir CACHE_DIR
                        Каталог для JSON/текста
  --log-dir LOG_DIR     Каталог для CSV‑логов
  --selftrain-dir SELFTRAIN_DIR
                        Каталог для self-supervised кропов (если не указан — не сохраняем)
  --init-batch INIT_BATCH
                        Начальный размер batch для TRBA
  --min-batch MIN_BATCH
                        Минимальный размер batch для TRBA перед fallback на CPU
  --gpu-index GPU_INDEX
                        Индекс GPU для логирования памяти
  --conf-high CONF_HIGH
                        Порог confidence для high_conf self-train
  --conf-mid CONF_MID   Порог для mid_conf self-train
  ```
