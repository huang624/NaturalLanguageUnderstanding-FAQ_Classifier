# NaturalLanguageUnderstanding-FAQ_Classifier
## Demo
![2022-05-22 18-30-41](https://user-images.githubusercontent.com/88367016/169691693-bf83fc7b-faf3-4a31-a015-a4ea5fffb29d.gif)
![2022-05-22 18-51-09](https://user-images.githubusercontent.com/88367016/169691802-ffa00b80-418d-46e3-a8d9-4293e722e186.gif)

## 介紹  
此模型為問題分類器，可將各式各樣的問題分類至台北市政府的各部門  

## 應用  
聊天機器人，輔助市民將問題導向相關單位

## Model
### Pretrain Model
```Python
from transformers import BertConfig, BertForSequenceClassification
config = BertConfig.from_pretrained('bert-base-chinese', num_labels=78)  
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
```
### Hyperparameters for Finetuning
Training_data:5397  
Evaluation_data:1350  
learning_rate=3e-5  
batch_size = 5  
epochs = 8  

### Metric
Accuarcy: 0.84
