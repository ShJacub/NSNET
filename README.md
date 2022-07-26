# Ссылка на оригинальный репозиторий

https://github.com/GuillaumeVW/NSNet.git

# Клонирование репозитория
```
git clone https://github.com/ShJacub/NSNet.git
```

# Создание и запуск docker-контейнера
```
docker create -v absolute_NSNet_directory:/NSNet -v absolute_path_to_dataset:/datasets --runtime=nvidia -it -p 1111:1111 --ipc=host --name=My_Docker floydhub/pytorch:1.3.0-gpu.cuda10cudnn7-py3.52 /bin/bash

docker start My_Docker

docker exec -it My_Docker /bin/bash
```

# Установка библиотек
```
cd /NSNet
pip install -r requirements.txt
```

# Перевод набора данных в wav формат
```
cd /NSNet

python MelToWav.py num_cpus
```

# Обучение
```
python train_nn.py num_workers
```

# Подсчёт метрики MSE
```
cd /NSNet

python my_test_nn.py path_to_weights

python MSE_calc.py num_cpus
```