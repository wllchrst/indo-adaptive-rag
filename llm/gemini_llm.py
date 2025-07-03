from interfaces import IDocument
from llm.base_llm import BaseLLM
from google import genai
from google.genai import types
from helpers import env_helper

class GeminiLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.API_KEY = env_helper.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.API_KEY)

    def answer(self, prompt: str):
        model = "gemini-2.0-flash-lite"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
        )

        result = ""

        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
            # print(chunk.text, end="")
        
        return result
    
    def format_with_document(self, prompt: str, documents: list[IDocument]) -> str:
        """
        Formats the prompt with the document content.
        """
        context_formatted = ""
        for doc in documents:
            context_formatted += f"Context Title: {doc.metadata.title}\n{doc.text}\n\n"
        
        return "\n".join([
            context_formatted,
            f"Q: {prompt}",
            "Berikan jawaban yang singkat."
        ])
    
    def validate_context(self, query: str, context:str) -> bool:
        question = f"Apakah konteks berikut cukup untuk menjawab pertanyaan: '{query}'?\n\nKonteks:\n{context}\n\nJawab dengan 'Ya' atau 'Tidak'."

        response = self.answer(question)
        print(response)
        
        return True if "Ya" in response else False