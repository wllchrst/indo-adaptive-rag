from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Classifier:
    def __init__(self,
                 model_path: str):
        self.tokenizer, self.model = self.gather_model(model_path)
        self.mapping = {
            0: "A",
            1: "B",
            2: "C",
        }

    def gather_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        return tokenizer, model

    def classify(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        predicted_class = outputs.logits.argmax(dim=-1).item()
        return self.mapping[predicted_class]
