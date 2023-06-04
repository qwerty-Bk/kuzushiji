# kuzushiji
Tsinghua, 2023 Spring, Deep Learning, Final project


How to use in Colab:
```
!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

api_token = {"username": "username", "key": "key"}

import json

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json
```

Download & unzip data:
``` 
!kaggle competitions download -c kuzushiji-recognition
!unzip kuzushiji-recognition.zip
%cd /content
!mkdir data
%cd data
!mkdir train
%cd train
!unzip /content/train_images.zip
%cd /content
%cd data
!mkdir test
%cd test
!unzip /content/test_images.zip
%cd /content
```

Font for Japanese:
```
!gdown --id 1prRT49D1yJFGo75oxpbYgPOfXnecKMtM
```
