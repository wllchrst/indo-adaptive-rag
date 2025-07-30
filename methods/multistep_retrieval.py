from typing import List
from methods.base_method import BaseMethod
from interfaces import IDocument
from helpers import WordHelper
class MultistepRetrieval(BaseMethod):
    def __init__(self, model_type: str):
        super().__init__(model_type)

    def answer(self, query: str, with_logging: bool = False, index: str=''):
        """
        This method retrieves multiple relevant documents from the vector database
        and uses them to answer the query.
        
        1. Retrieve document
        2. Do Reasoning from previous document and reasoning to ask for better query in the next iteration
        3. Repeat until this things is happening
           a. The context answer has answer inside it then stops
           b. Limitations of reptitions in the retrieval is reached

        """
        retrieved_documents = self.retrieve(
            query, index=index, with_logging=with_logging)

        return retrieved_documents
        
    def retrieve(self,
                original_question: str,
                previous_reasonings: List[str] = [],
                previous_documents: List[IDocument] = [],
                query: str = '',
                current_count: int = 0,
                limit_count: int = 5,
                index: str ='',
                retrieval_count: int = 3,
                with_logging: bool = False):
        retrieval_query = original_question if query == '' else query
        documents = self.retrieve_document(retrieval_query, total_result=retrieval_count, index=index)

        result, is_answered = self.reasoning(
            question=original_question,
            documents=documents,
            previous_reasonings=previous_reasonings
        )

        if is_answered:
            return result
        elif current_count >= limit_count:
            return "Tidak dapat menemukan jawaban yang tepat. Silakan coba pertanyaan lain."

        previous_reasonings.append(result)
        previous_documents.append(documents)
        
        self.retrieve(
            original_question=original_question,
            query=result,
            limit_count=limit_count,
            current_count=current_count + 1,
            previous_reasonings=previous_reasonings,
            previous_documents=previous_documents,
            with_logging=with_logging
        )

    def reasoning(self,
                question: str,
                documents: List[IDocument],
                previous_reasonings: List[str],
                with_logging: bool = False):
        context = "\n\n".join(
            [
                "Context Title: "
                + doc.metadata.title
                + "\n"
                + doc.text.strip().replace("\n", " ").strip()
                for doc in documents
            ]
        ).strip()

        qn_pretext = "Q: "
        question_instruction = "Jawablah pertanyaan berikut dengan penalaran langkah demi langkah.\n"
        answer_instruction = "Jika informasi yang diberikan tidak cukup untuk menjawab pertanyaan, berikan saja kata kunci yang dapat digunakan untuk menjawab pertanyaan. Jika informasi yang diberikan cukup, berikan jawaban yang tepat.\n\nJawaban:"
        
        an_pretext = "A: "
        reasoning_joined = " ".join(previous_reasonings)
        
        reasoning_prompt = "\n".join([
            context, "", f"{qn_pretext} {question_instruction}{question}", f"{an_pretext} {reasoning_joined}", answer_instruction
        ])

        if with_logging:
            print(f'Reasoning prompt: {reasoning_prompt}')

        answer = self.llm.answer(reasoning_prompt)
        answer = WordHelper.clean_sentence(answer)

        if "Jawaban" in answer:
            answer = answer.split("Jawaban")[1].strip()
            return WordHelper.remove_non_alphabetic(answer).strip(), True
        
        return answer, False