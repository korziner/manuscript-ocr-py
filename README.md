# manuscript-ocr-py
Минималистичная CLI-распознавалка моделями konstantinkozhin/manuscript-ocr

py 3.14 tested
<img width="1341" height="740" alt="image" src="https://github.com/user-attachments/assets/51b72a7a-3835-4170-9d86-9193507cfee8" />

💥 OOM at batch 147. New safe max = 102:
```
2026-06-07 07:26:25,438 INFO 📈 Increasing batch size: 103 → 123
2026-06-07 07:26:25,439 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p288_cb65.jpg] TRBA: 4.149s, final_batch=103, reductions=0, fallback=False
2026-06-07 07:26:25,592 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p288_cb65.jpg] Total: 6.137s, 159 words, 25.91 words/sec
2026-06-07 07:26:25,595 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p289_7547.jpg] running EAST.
2026-06-07 07:26:27,149 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p289_7547.jpg] EAST: 1.546s
2026-06-07 07:26:27,397 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p289_7547.jpg] TRBA: 166 words, current batch=123.
2026-06-07 07:26:31,661 INFO 📈 Increasing batch size: 123 → 147
2026-06-07 07:26:31,662 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p289_7547.jpg] TRBA: 4.264s, final_batch=123, reductions=0, fallback=False
2026-06-07 07:26:31,796 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p289_7547.jpg] Total: 6.201s, 166 words, 26.77 words/sec
2026-06-07 07:26:31,798 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p290_284a.jpg] running EAST.
2026-06-07 07:26:33,201 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p290_284a.jpg] EAST: 1.396s
2026-06-07 07:26:33,398 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p290_284a.jpg] TRBA: 141 words, current batch=147.
2026-06-07 07:26:36,853 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p290_284a.jpg] TRBA: 3.454s, final_batch=147, reductions=0, fallback=False
2026-06-07 07:26:36,921 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p290_284a.jpg] Total: 5.122s, 141 words, 27.53 words/sec
2026-06-07 07:26:36,922 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p291_f507.jpg] running EAST.
2026-06-07 07:26:39,082 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p291_f507.jpg] EAST: 2.098s
2026-06-07 07:26:39,330 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p291_f507.jpg] TRBA: 186 words, current batch=147.
2026-06-07 07:26:39.493748329 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned while running Conv node. Name:'/cnn/conv0/conv0.3/Conv' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 4531027968

2026-06-07 07:26:39,494 WARNING OOM at batch 147. Reducing.
2026-06-07 07:26:39,494 WARNING 💥 OOM at batch 147. New safe max = 102, reducing to 102
2026-06-07 07:26:44,350 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p291_f507.jpg] TRBA: 5.019s, final_batch=102, reductions=1, fallback=False
2026-06-07 07:26:44,422 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p291_f507.jpg] Total: 7.499s, 186 words, 24.80 words/sec
2026-06-07 07:26:44,422 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p292_36f2.jpg] running EAST.
2026-06-07 07:26:45,800 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p292_36f2.jpg] EAST: 1.372s
2026-06-07 07:26:46,011 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p292_36f2.jpg] TRBA: 141 words, current batch=102.
2026-06-07 07:26:49,786 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p292_36f2.jpg] TRBA: 3.774s, final_batch=102, reductions=0, fallback=False
2026-06-07 07:26:49,859 INFO [Указы_Сибирской_губернской_канцелярии_выписки_из_журналов_воеводской_канцелярии_рапорта_офицеров_о_с_p292_36f2.jpg] Total: 5.435s, 141 words, 25.94 words/sec
```


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
