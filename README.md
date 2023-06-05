# kuzushiji
Tsinghua, 2023 Spring, Deep Learning, Final project


### How to use in Colab

Install library for logging:
``` 
!pip install wandb --quiet
```

Download dataset:
```
!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

api_token = {"username": "username", "key": "key"}

import json

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json
```

Clone github repo:
```
!git clone https://github.com/qwerty-Bk/kuzushiji.git
```

Download & unzip data:
``` 
%cd kuzushiji
!kaggle competitions download -c kuzushiji-recognition
!mkdir data
%cd data
!unzip /content/kuzushiji/kuzushiji-recognition.zip
!mkdir train
%cd train
!unzip /content/kuzushiji/train_images.zip
%cd /content/kuzushiji/
%cd data
!mkdir test
%cd test
!unzip /content/kuzushiji/test_images.zip
%cd /content/kuzushiji
```

Font for Japanese:
```
%cd ..
!gdown --id 1prRT49D1yJFGo75oxpbYgPOfXnecKMtM
%cd /content/kuzushiji
```
