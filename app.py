from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification, default_data_collator
from accelerate import Accelerator
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import warnings
from sklearn.preprocessing import LabelEncoder
import pickle
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


class Taipei_FAQ_Classifier_Request(BaseModel):
    question: constr(max_length=256)

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

def Taipei_FAQ_Classifier_predict(model, question):
    input_encodings = tokenizer([question], truncation=True, padding=True)
    input_dataset = Dataset(input_encodings)
    data_collator = default_data_collator
    input_dataloader = DataLoader(input_dataset, collate_fn=data_collator, batch_size=1)  
  
    accelerator = Accelerator()
    model, input_dataloader = accelerator.prepare(model, input_dataloader)
  
    for batch in input_dataloader:
      outputs = model(**batch)
      predicted = outputs.logits.argmax(dim=-1)
    return predicted.item()


app = FastAPI(
    title="Taipei_FAQ_Classifier",
    description="Taipei_FAQ_Classifier dataset training",
    version="1",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open('label_encoder.pkl', 'rb') as reader:
    le = pickle.load(reader)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
config = BertConfig.from_pretrained("Taipei_FAQ_Classifier_config.json") 
Taipei_FAQ_Classifier_model = BertForSequenceClassification.from_pretrained("Taipei_FAQ_Classifier_model.bin", config = config)

@app.get("/")
async def root():
    return RedirectResponse("docs")


@app.get("/page/{page_name}", response_class=HTMLResponse)
async def page(request: Request, page_name: str):
    return templates.TemplateResponse(f"{page_name}.html", {"request": request})


@app.post("/Taipei_FAQ_Classifier")
async def Taipei_FAQ_Classifier(
    Taipei_FAQ_Classifier_request: Taipei_FAQ_Classifier_Request = Body(
        None,
        
    )
):
    

    results = Taipei_FAQ_Classifier_predict(Taipei_FAQ_Classifier_model, Taipei_FAQ_Classifier_request.question)

    result = le.inverse_transform([results])[0]
    return "處理單位:"+result

