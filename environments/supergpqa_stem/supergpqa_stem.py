from enum import Enum
from typing import Dict

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()

ZERO_SHOT_PROMPT_TEMPLATE = """
Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: \\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

{}
""".strip()

FIVE_SHOT_PROMPT_TEMPLATE = """
Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: \\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question:
A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is
A) 10
B) 40
C) 6
D) 25
E) 15
F) 50
G) 30
H) 4
I) 5
J) 20

Answer: Let's think step by step. In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4.
Answer: \\boxed{{H}}.

Question:
Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?
A) 1000 times more
B) 50 times more
C) 5000 times more
D) 500 times more
E) 10000 times more
F) 20000 times more
G) 2000 times more
H) 100 times more
I) 10 times more
J) N/A

Answer: Let's think step by step. The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula $A = \\pi \\left(\\frac{{D}}{{2}}\\right)^2$, where $D$ is the diameter. Therefore, the relative light-gathering power is calculated as:
\\[
\\frac{{\\left(\\frac{{50 \\text{{ cm}}}}{{2}}\\right)^2}}{{\\left(\\frac{{5 \\text{{ mm}}}}{{2}}\\right)^2}} = \\frac{{\\left(\\frac{{50 \\text{{ cm}}}}{{0.1 \\text{{ cm}}}}\\right)^2}}{{\\left(\\frac{{5 \\text{{ mm}}}}{{0.1 \\text{{ cm}}}}\\right)^2}} = \\frac{{500^2}}{{5^2}} = 10000.
\\]
Answer: \\boxed{{E}}.

Question:
Where do most short-period comets come from and how do we know?
A) The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt.
B) The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.
C) The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.
D) The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.
E) The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.
F) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.
G) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.

Answer: Let's think step by step. Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin.
Answer: \\boxed{{A}}.

Question:
Colors in a soap bubble result from light
A) dispersion
B) deflection
C) refraction
D) reflection
E) interference
F) converted to a different frequency
G) polarization
H) absorption
I) diffraction
J) transmission

Answer: Let's think step by step. The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light.
Answer: \\boxed{{E}}.

Question:
A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A) 240 W
B) 120 W
C) 10 W
D) 480 W
E) 360 W
F) 200 W
G) 30 W
H) 150 W
I) 60 W
J) 300 W

Answer: Let's think step by step. The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:
\\[
\\text{{Power}} = \\text{{Voltage}} \\times \\text{{Current}} = 120 \\, \\text{{V}} \\times 2 \\, \\text{{A}} = 240 \\, \\text{{W}}.
\\]
Therefore, the microwave oven uses energy at a rate of 240 watts.
Answer: \\boxed{{A}}.

Question:
{}

Answer: Let's think step by step.
""".strip()

STEM_DISCIPLINES = {"Science", "Engineering"}


class Difficulty(str, Enum):
    ALL = "all"
    EASY = "easy"
    MIDDLE = "middle"
    HARD = "hard"


def _build_question(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in options.items() if v not in [None, ""])
    return f"{question}\n{opts}"


def _to_vf_format(
    ds: Dataset,
    few_shot: bool,
    shuffle_answers: bool,
    shuffle_seed: int | None,
) -> Dataset:
    VALID = "ABCDEFGHIJ"
    prompt_template = FIVE_SHOT_PROMPT_TEMPLATE if few_shot else ZERO_SHOT_PROMPT_TEMPLATE

    def _format_row(row: dict, idx: int) -> dict:
        question = row.get("question", "") or ""
        opts = row.get("options", {}) or {}
        opts = {k: v for k, v in opts.items() if v not in [None, ""]}

        answer_letter = (row.get("answer_letter") or "").strip().upper()
        if answer_letter not in VALID:
            return None

        if shuffle_answers and answer_letter and answer_letter in opts:
            opts, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=idx,
            )

        question_prompt = _build_question(question, opts)
        prompt = prompt_template.format(question_prompt)

        info = dict(row)
        if shuffle_answers:
            info["answer"] = answer_letter
            info["options"] = opts
        info["answer_text"] = opts.get(answer_letter, None)

        return {"question": prompt, "answer": answer_letter, "info": info}

    return ds.map(_format_row, remove_columns=ds.column_names, load_from_cache_file=False, with_indices=True)


def load_environment(
    disciplines: list[str] | None = None,
    field: str | None = None,
    difficulty: str | Difficulty = Difficulty.ALL,
    few_shot: bool = False,
    shuffle_answers: bool = True,
    shuffle_seed: int | None = 1618,
    test_size: float = 0.1,
    split_seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """
    Single-turn STEM RL environment using the non-medicine sections of m-a-p/SuperGPQA.

    Args:
        disciplines: List of disciplines to include. Defaults to ['Science', 'Engineering'].
        field: Filter by field within a discipline (e.g. 'Physics', 'Chemistry'). None = all fields.
        difficulty: Filter by difficulty ('easy', 'middle', 'hard', 'all').
        few_shot: Include 5-shot examples in prompts.
        shuffle_answers: Shuffle answer choices to prevent positional bias. Defaults to True.
        shuffle_seed: Seed for deterministic answer shuffling.
        test_size: Fraction of data to use as eval split (default 0.1).
        split_seed: Seed for train/eval split (default 42).
    """
    if disciplines is None:
        disciplines = list(STEM_DISCIPLINES)

    raw = load_dataset("m-a-p/SuperGPQA", split="train").filter(
        lambda row: row["discipline"] in disciplines
    )

    if field is not None:
        raw = raw.filter(lambda row: row["field"].lower() == field.lower())

    difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
    if difficulty != Difficulty.ALL:
        raw = raw.filter(lambda row: row["difficulty"] == difficulty.value)

    # Convert options from list to dict with letter keys
    def _convert_options(row: dict) -> dict:
        opts = row["options"]
        if isinstance(opts, list):
            row["options"] = {chr(ord("A") + i): v for i, v in enumerate(opts)}
        return row

    raw = raw.map(_convert_options, load_from_cache_file=False)

    # Carve out train/eval split (SuperGPQA only has a single "train" split)
    splits = raw.train_test_split(test_size=test_size, seed=split_seed)
    train_raw = splits["train"]
    test_raw = splits["test"]

    train_ds = _to_vf_format(train_raw, few_shot=few_shot, shuffle_answers=shuffle_answers, shuffle_seed=shuffle_seed)
    test_ds = _to_vf_format(test_raw, few_shot=few_shot, shuffle_answers=shuffle_answers, shuffle_seed=shuffle_seed)

    del train_raw, test_raw

    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=test_ds,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
