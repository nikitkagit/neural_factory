from __future__ import annotations
from typing import List, Tuple, Union, Optional


class Prompt:
    """
    A composable prompt builder.

    Each Prompt instance stores a list of (priority, text) pieces.
    When converted to a string, the pieces are sorted by priority
    (lower number = earlier in the final text) and joined with ". ".
    """

    def __init__(
        self,
        text: str = "",
        priority: Union[int, float] = 0,
        _pieces: Optional[List[Tuple[Union[int, float], str]]] = None,
    ) -> None:
        """
        Initialize a Prompt.

        :param text: The text of this prompt piece.
        :param priority: Sorting priority (lower = earlier in output).
        :param _pieces: Internal list used when combining prompts.
        """
        if _pieces is not None:
            self.pieces: List[Tuple[Union[int, float], str]] = list(_pieces)
        else:
            self.pieces = [(priority, text)]

    def __add__(self, other: Union["Prompt", str]) -> "Prompt":
        """Handle Prompt + Prompt and Prompt + str."""
        if isinstance(other, Prompt):
            return Prompt(_pieces=self.pieces + other.pieces)
        if isinstance(other, str):
            # Plain strings are appended at the very end.
            return Prompt(_pieces=self.pieces + [(float("inf"), other)])
        return NotImplemented

    def __radd__(self, other: Union["Prompt", str]) -> "Prompt":
        """Handle str + Prompt (left-side string concatenation)."""
        if isinstance(other, str):
            # A string on the left gets the highest priority (earliest position).
            return Prompt(_pieces=[(float("-inf"), other)] + self.pieces)
        return NotImplemented

    def __str__(self) -> str:
        """Render the combined prompt as a single string."""
        # 1. Sort pieces by priority.
        sorted_pieces = sorted(self.pieces, key=lambda item: item[0])

        # 2. Clean each piece: strip whitespace and trailing dots.
        cleaned_texts = [
            text.strip().rstrip(".")
            for _, text in sorted_pieces
            if text and text.strip()
        ]

        # 3. Join with ". " to form a single sentence.
        return ". ".join(cleaned_texts)

    def __repr__(self) -> str:
        return f"Prompt({str(self)!r})"

    def __bool__(self) -> bool:
        """A Prompt is truthy if it contains any non-empty text."""
        return any(text.strip() for _, text in self.pieces)



# ---------------------------------------------------------------------------
# Base prompts
# ---------------------------------------------------------------------------


BE_SHORT = Prompt(
    "Respond as briefly and to the point as possible",
    priority=100
)


PYTHON_CODE_ONLY = Prompt(
    "Output ONLY clean Python code. No explanations, no markdown wrappers (```), no comments outside the code. Only the executable code itself",
    priority=100
)
NO_CODE = Prompt(
    "Do not write any code.",
    priority=100
)
CODE_COMMENTS_ENGLISH = Prompt(
    "Comment code in English.",
    priority=100
)


GEN_SPEC = Prompt(
    """
    Generate a specification by the Main request in Russian.
    The specification MUST have a strict structure with these sections:
    ## Цель [The overall goal of the changes. Contains a reformulated and more rigorously described main request.]
    ## Необходимые файлы [which contain the information necessary for implementation. here you only need to provide a list of files in the format without - symbol before it: path/to/file.code.]
    ## Изменяемые файлы [which will need to be edited. Similar format.]
    ## Изменения
    ### <file path>:
        <list of changes>
    ## Тесты [semantics that one needs to know in order to subsequently write tests]
    Use bullet points, not paragraphs. No general reasoning.
    Always include exact method/class names.
    """,
    priority=100
)
GEN_SPEC_FOR_TESTS = Prompt(
    """
    Generate a specification for tests by the given specification in Russian.
    The specification MUST have a strict structure with these sections:
    ## Цель [The short overall goal of the changes.]
    ## Необходимые файлы [which contain the information necessary for implementation. here you only need to provide a list of files in the format WITHOUT MINUS '-' symbol before it: path/to/file.code.]
    ## Изменяемые файлы [which will need to be edited. Similar format.]
    ## Изменения
    ### <file path>:
        <list of changes>
    No general reasoning. Always include exact method/class names. You mustn't use minus '-' before filepaths.
    """,
    priority=100
)


PATHS_RELATIVE_TO_ROOT = Prompt(
    "All file paths provided or requested are strictly relative to the project root directory.",
    priority=100
)


def make_request_prompt(request: str, priority: int = 500) -> Prompt:
    """Creates a prompt with user name context."""
    return Prompt(f"Main request: {request}", priority=priority)

def make_code_prompt(code: str, priority: int = 1000) -> Prompt:
    return Prompt(f"Some code of the project: {code}", priority=priority)

def make_specification_prompt(spec: str, priority: int = 1000) -> Prompt:
    return Prompt(f"Main specification: {spec}", priority=priority)

def make_previous_specification_prompt(spec: str, priority: int = 1000) -> Prompt:
    return Prompt(f"Previous specification: {spec}", priority=priority)