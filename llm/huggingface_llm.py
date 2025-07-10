from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.base_llm import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str = 'cahya/gpt2-small-indonesian-522M'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def answer(self, query: str):
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer