from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_id)
model = DistilBertForSequenceClassification.from_pretrained(model_id).eval()


examples = [
    "Hello, world!",
    "Nice to meet you",
    "Goodbye, world!"
]
inputs = tokenizer(examples, return_tensors="pt")


def run_model(inputs):
    return model(**inputs)


for i in range(3):
    result = run_model(inputs)

predicted_class_ids = result.logits.argmax(dim=1)

for id in predicted_class_ids.numpy():
    print(model.config.id2label[id])
