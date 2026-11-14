"""Generator that only answers from the supplied documents."""

from rag_titan.generation.base_generator import BaseGenerator
from rag_titan.prompts import GROUNDED_ANSWER_PROMPT, NO_ANSWER


class GroundedGenerator(BaseGenerator):
    """
    Answers only from the given chunks and cites their ids.

    Args:
        llm: any BaseLLM.
        instructions: system prompt, defaults to GROUNDED_ANSWER_PROMPT.
    """

    def __init__(self, llm, instructions=GROUNDED_ANSWER_PROMPT):
        super().__init__(llm, instructions)

    def generate(self, query, chunks):
        """Return an answer with citations, or NO_ANSWER when unsure."""
        if not chunks:
            return NO_ANSWER

        user_input = self._build_user_input(query, chunks)
        answer = self.llm.ask(self.instructions, user_input)

        return answer or NO_ANSWER
