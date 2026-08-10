from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


DATASET_HANDLE = "thedevastator/sciq-a-dataset-for-science-question-answering"
REQUIRED_COLUMNS = {
    "question",
    "correct_answer",
    "distractor1",
    "distractor2",
    "distractor3",
    "support",
}
CHOICE_COLUMNS = ["correct_answer", "distractor1", "distractor2", "distractor3"]


def _normal_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def clean_sciq_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a SciQ split and validate the expected fields."""
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    result = result.loc[:, ~result.columns.str.startswith("unnamed")]

    missing = REQUIRED_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"The SciQ file is missing required columns: {sorted(missing)}")

    for column in REQUIRED_COLUMNS:
        result[column] = result[column].map(_normal_text)

    result = result[result["question"].str.len() > 0]
    result = result[result["correct_answer"].str.len() > 0]
    result = result.drop_duplicates(subset=["question", "correct_answer"]).reset_index(drop=True)
    return result


def _find_split_file(search_roots: Sequence[Path], names: Sequence[str]) -> Optional[Path]:
    lowered = {name.lower() for name in names}
    for root in search_roots:
        if not root.exists():
            continue
        for file_path in root.rglob("*.csv"):
            if file_path.name.lower() in lowered:
                return file_path
    return None


def download_sciq_from_kaggle(output_dir: str | Path = "data") -> Dict[str, Path]:
    """Download the public SciQ dataset from Kaggle and locate its three CSV files."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(output_path))
    except TypeError:
        downloaded = kagglehub.dataset_download(DATASET_HANDLE)
    except Exception as exc:
        raise RuntimeError(
            "Kaggle could not download the public SciQ dataset. Check the internet connection. "
            "If Kaggle requests authentication, add your Kaggle API token and run the cell again."
        ) from exc

    search_roots = [Path(downloaded), output_path]
    split_names = {
        "train": ["train.csv"],
        "validation": ["validation.csv", "valid.csv", "val.csv"],
        "test": ["test.csv"],
    }
    files: Dict[str, Path] = {}
    for split, possible_names in split_names.items():
        found = _find_split_file(search_roots, possible_names)
        if found is None:
            available = sorted(
                str(path) for root in search_roots if root.exists() for path in root.rglob("*.csv")
            )
            raise FileNotFoundError(
                f"Could not find the {split} CSV file. CSV files found: {available}"
            )
        files[split] = found
    return files


def load_sciq_data(output_dir: str | Path = "data") -> Dict[str, pd.DataFrame]:
    files = download_sciq_from_kaggle(output_dir)
    return {split: clean_sciq_frame(pd.read_csv(path)) for split, path in files.items()}


@dataclass
class SearchResult:
    row_index: int
    score: float
    question: str
    answer: str
    support: str


class ScienceKnowledgeBase:
    """TF IDF knowledge index used to search SciQ questions and support passages."""

    def __init__(self, max_features: int = 50000) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.frame: Optional[pd.DataFrame] = None
        self.matrix: Optional[csr_matrix] = None

    def fit(self, frame: pd.DataFrame) -> "ScienceKnowledgeBase":
        self.frame = clean_sciq_frame(frame)
        documents = (
            self.frame["question"]
            + " "
            + self.frame["support"]
            + " "
            + self.frame["correct_answer"]
        ).tolist()
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.frame is None or self.matrix is None:
            raise RuntimeError("The knowledge base must be fitted before search is used.")
        query = _normal_text(query)
        if not query:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        if len(scores) == 0:
            return []
        top_k = max(1, min(int(top_k), len(scores)))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        results: List[SearchResult] = []
        for index in indices:
            row = self.frame.iloc[int(index)]
            results.append(
                SearchResult(
                    row_index=int(index),
                    score=float(scores[index]),
                    question=row["question"],
                    answer=row["correct_answer"],
                    support=row["support"],
                )
            )
        return results


class CandidateAnswerModel:
    """A supervised model that scores four SciQ answer choices for each question."""

    def __init__(self, random_state: int = 42, max_features: int = 45000) -> None:
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self.classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.is_fitted = False

    @staticmethod
    def _choice_rows(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        rng = random.Random(seed)
        records: List[Dict[str, Any]] = []
        for question_id, row in frame.reset_index(drop=True).iterrows():
            choices = [
                (row["correct_answer"], 1),
                (row["distractor1"], 0),
                (row["distractor2"], 0),
                (row["distractor3"], 0),
            ]
            rng.shuffle(choices)
            for position, (candidate, label) in enumerate(choices):
                records.append(
                    {
                        "question_id": int(question_id),
                        "position": int(position),
                        "question": row["question"],
                        "support": row["support"],
                        "candidate": candidate,
                        "label": int(label),
                    }
                )
        return pd.DataFrame.from_records(records)

    def _numeric_features(self, candidates: pd.DataFrame) -> np.ndarray:
        questions = candidates["question"].tolist()
        supports = candidates["support"].tolist()
        answers = candidates["candidate"].tolist()

        question_vectors = self.vectorizer.transform(questions)
        support_vectors = self.vectorizer.transform(supports)
        answer_vectors = self.vectorizer.transform(answers)

        question_similarity = np.asarray(
            question_vectors.multiply(answer_vectors).sum(axis=1)
        ).ravel()
        support_similarity = np.asarray(
            support_vectors.multiply(answer_vectors).sum(axis=1)
        ).ravel()

        features: List[List[float]] = []
        for index, (question, support, answer) in enumerate(zip(questions, supports, answers)):
            answer_tokens = set(_tokenize(answer))
            question_tokens = set(_tokenize(question))
            support_tokens = set(_tokenize(support))
            token_count = max(1, len(answer_tokens))
            exact_support = float(answer.lower() in support.lower() and bool(answer.strip()))
            support_coverage = len(answer_tokens.intersection(support_tokens)) / token_count
            question_coverage = len(answer_tokens.intersection(question_tokens)) / token_count
            answer_word_count = min(len(_tokenize(answer)), 15) / 15.0
            has_support = float(bool(support.strip()))
            numeric_answer = float(any(token.isdigit() for token in answer_tokens))
            features.append(
                [
                    float(support_similarity[index]),
                    float(question_similarity[index]),
                    exact_support,
                    float(support_coverage),
                    float(question_coverage),
                    float(answer_word_count),
                    has_support,
                    numeric_answer,
                ]
            )
        return np.asarray(features, dtype=np.float64)

    def fit(self, train_frame: pd.DataFrame) -> "CandidateAnswerModel":
        train_frame = clean_sciq_frame(train_frame)
        candidate_text = []
        candidate_text.extend(train_frame["question"].tolist())
        candidate_text.extend(train_frame["support"].tolist())
        for column in CHOICE_COLUMNS:
            candidate_text.extend(train_frame[column].tolist())
        self.vectorizer.fit(candidate_text)

        candidates = self._choice_rows(train_frame, self.random_state)
        x_train = self._numeric_features(candidates)
        y_train = candidates["label"].to_numpy()
        self.classifier.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict_choice_table(self, frame: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("The answer model must be fitted before prediction.")
        clean_frame = clean_sciq_frame(frame)
        candidates = self._choice_rows(clean_frame, seed)
        x_values = self._numeric_features(candidates)
        candidates = candidates.copy()
        candidates["probability"] = self.classifier.predict_proba(x_values)[:, 1]
        return candidates

    def evaluate(self, test_frame: pd.DataFrame) -> Dict[str, Any]:
        candidates = self.predict_choice_table(test_frame)
        selected = candidates.loc[candidates.groupby("question_id")["probability"].idxmax()]
        y_true = np.ones(len(selected), dtype=int)
        y_pred = selected["label"].to_numpy()
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "multiple_choice_accuracy": float(accuracy),
            "questions_evaluated": int(len(selected)),
            "correct_predictions": int(y_pred.sum()),
        }

    def predict_row(self, row: pd.Series, seed: int = 99) -> Dict[str, Any]:
        one_row = pd.DataFrame([row])
        table = self.predict_choice_table(one_row, seed=seed).sort_values(
            "probability", ascending=False
        )
        best = table.iloc[0]
        return {
            "answer": best["candidate"],
            "probability": float(best["probability"]),
            "is_correct": bool(best["label"]),
            "ranked_choices": table[["candidate", "probability"]].to_dict("records"),
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> "CandidateAnswerModel":
        model = joblib.load(Path(path))
        if not isinstance(model, CandidateAnswerModel):
            raise TypeError("The saved object is not a CandidateAnswerModel.")
        return model


@dataclass
class AgentState:
    current_quiz: Optional[Dict[str, Any]] = None
    correct: int = 0
    attempted: int = 0
    recent_topics: List[str] = field(default_factory=list)


class AIStudyAssistantAgent:
    """A retrieval based study assistant with question answering, quizzes, and tips."""

    QUIZ_WORDS = ("quiz", "test me", "practice question", "practice quiz")
    TIP_WORDS = ("study tip", "study plan", "how should i study", "help me study")
    HELP_WORDS = ("help", "commands", "what can you do")

    def __init__(
        self,
        knowledge_base: ScienceKnowledgeBase,
        source_frame: pd.DataFrame,
        confidence_threshold: float = 0.08,
        random_state: int = 42,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.source_frame = clean_sciq_frame(source_frame)
        self.confidence_threshold = confidence_threshold
        self.rng = random.Random(random_state)
        self.state = AgentState()

    def reset(self) -> str:
        self.state = AgentState()
        return "Your quiz score and current activity have been reset."

    def route_intent(self, message: str) -> str:
        text = message.lower().strip()
        if self.state.current_quiz and self._parse_choice(message) is not None:
            return "quiz_answer"
        if any(word in text for word in self.QUIZ_WORDS):
            return "quiz"
        if any(word in text for word in self.TIP_WORDS):
            return "study_tips"
        if any(word == text or word in text for word in self.HELP_WORDS):
            return "help"
        if text in {"reset", "start over", "clear score"}:
            return "reset"
        return "question"

    @staticmethod
    def _parse_choice(message: str) -> Optional[int]:
        text = message.strip().lower()
        match = re.search(r"\b([abcd])\b", text)
        if match:
            return ord(match.group(1)) - ord("a")
        match = re.search(r"\b([1-4])\b", text)
        if match:
            return int(match.group(1)) - 1
        return None

    @staticmethod
    def _short_explanation(support: str, answer: str, maximum_sentences: int = 2) -> str:
        support = _normal_text(support)
        if not support:
            return "The dataset does not include a support passage for this item."
        sentences = re.split(r"(?<=[.!?])\s+", support)
        answer_lower = answer.lower()
        selected = [sentence for sentence in sentences if answer_lower in sentence.lower()]
        if not selected:
            selected = sentences[:maximum_sentences]
        return " ".join(selected[:maximum_sentences]).strip()

    @staticmethod
    def _extract_topic(message: str) -> str:
        cleaned = message.lower()
        for phrase in AIStudyAssistantAgent.QUIZ_WORDS + AIStudyAssistantAgent.TIP_WORDS:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"\b(on|about|for|please|a|an|the|me|make|create|give|build)\b", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def answer_question(self, message: str) -> str:
        results = self.knowledge_base.search(message, top_k=3)
        if not results or results[0].score < self.confidence_threshold:
            return (
                "I could not find a strong match in the SciQ science material. "
                "Try using a more specific science term or ask for a quiz on biology, chemistry, physics, earth science, or astronomy."
            )
        best = results[0]
        explanation = self._short_explanation(best.support, best.answer)
        related = ""
        if len(results) > 1 and results[1].score >= self.confidence_threshold:
            related = f"\n\nA related question in the dataset is: {results[1].question}"
        return (
            f"Answer: {best.answer}\n\n"
            f"Explanation: {explanation}\n\n"
            f"Closest study question: {best.question}\n"
            f"Search confidence: {best.score:.2f}"
            f"{related}"
        )

    def start_quiz(self, message: str = "") -> str:
        topic = self._extract_topic(message)
        if topic:
            matches = self.knowledge_base.search(topic, top_k=min(40, len(self.source_frame)))
            useful = [match for match in matches if match.score > 0]
            if useful:
                selected_match = self.rng.choice(useful[: min(15, len(useful))])
                row = self.source_frame.iloc[selected_match.row_index]
            else:
                row = self.source_frame.sample(1, random_state=self.rng.randint(0, 1_000_000)).iloc[0]
        else:
            row = self.source_frame.sample(1, random_state=self.rng.randint(0, 1_000_000)).iloc[0]

        choices = [row[column] for column in CHOICE_COLUMNS]
        self.rng.shuffle(choices)
        correct_index = choices.index(row["correct_answer"])
        self.state.current_quiz = {
            "question": row["question"],
            "choices": choices,
            "correct_index": correct_index,
            "correct_answer": row["correct_answer"],
            "support": row["support"],
            "topic": topic or "general science",
        }
        self.state.recent_topics.append(topic or "general science")
        self.state.recent_topics = self.state.recent_topics[-10:]

        choice_text = "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
        )
        return (
            f"Practice question on {topic or 'general science'}:\n\n"
            f"{row['question']}\n\n{choice_text}\n\n"
            "Reply with A, B, C, or D."
        )

    def grade_quiz(self, message: str) -> str:
        if not self.state.current_quiz:
            return "There is no active quiz question. Type quiz to begin."
        selected_index = self._parse_choice(message)
        if selected_index is None:
            return "Please answer the quiz with A, B, C, or D."

        quiz = self.state.current_quiz
        self.state.attempted += 1
        is_correct = selected_index == quiz["correct_index"]
        if is_correct:
            self.state.correct += 1
            result = "Correct."
        else:
            selected_text = quiz["choices"][selected_index]
            result = (
                f"That answer was {selected_text}. The correct answer is {quiz['correct_answer']}."
            )
        explanation = self._short_explanation(quiz["support"], quiz["correct_answer"])
        score = f"Score: {self.state.correct} out of {self.state.attempted}."
        self.state.current_quiz = None
        return f"{result}\n\nExplanation: {explanation}\n\n{score}\n\nType quiz for another question."

    def study_tips(self, message: str = "") -> str:
        topic = self._extract_topic(message) or (
            self.state.recent_topics[-1] if self.state.recent_topics else "science"
        )
        if self.state.attempted == 0:
            performance = "You have not completed a quiz yet, so begin with a short practice set."
        else:
            rate = self.state.correct / self.state.attempted
            if rate < 0.5:
                performance = "Your current quiz score suggests that you should review the basic ideas before taking another quiz."
            elif rate < 0.8:
                performance = "Your current quiz score shows progress. Review each missed answer and then try another set."
            else:
                performance = "Your current quiz score is strong. Use harder questions and explain each answer in your own words."
        return (
            f"Study plan for {topic}:\n\n"
            f"1. Spend ten minutes reviewing the main terms.\n"
            f"2. Ask the agent two questions about ideas that are unclear.\n"
            f"3. Complete five practice questions without looking at notes.\n"
            f"4. Review every missed answer and write one sentence explaining it.\n"
            f"5. Take a short break, then repeat the hardest questions.\n\n"
            f"Personal note: {performance}"
        )

    @staticmethod
    def help_text() -> str:
        return (
            "You can use the assistant in four main ways:\n\n"
            "1. Ask a science question, such as What is photosynthesis?\n"
            "2. Ask for an explanation, such as Explain kinetic energy.\n"
            "3. Start a quiz by typing quiz or quiz me on biology.\n"
            "4. Ask for study tips or a study plan.\n\n"
            "Type reset to clear your quiz score."
        )

    def respond(self, message: str) -> str:
        message = _normal_text(message)
        if not message:
            return "Please enter a question or type help."
        intent = self.route_intent(message)
        if intent == "quiz_answer":
            return self.grade_quiz(message)
        if intent == "quiz":
            return self.start_quiz(message)
        if intent == "study_tips":
            return self.study_tips(message)
        if intent == "help":
            return self.help_text()
        if intent == "reset":
            return self.reset()
        return self.answer_question(message)


def evaluate_intent_router(agent: AIStudyAssistantAgent) -> Dict[str, Any]:
    tests = [
        ("quiz me on biology", "quiz"),
        ("give me a practice question", "quiz"),
        ("make a study plan for chemistry", "study_tips"),
        ("what can you do", "help"),
        ("What is kinetic energy?", "question"),
        ("reset", "reset"),
    ]
    predictions = [agent.route_intent(text) for text, _ in tests]
    expected = [label for _, label in tests]
    correct = sum(predicted == label for predicted, label in zip(predictions, expected))
    return {
        "intent_accuracy": correct / len(tests),
        "intent_tests": len(tests),
        "intent_details": [
            {"input": text, "expected": label, "predicted": predicted}
            for (text, label), predicted in zip(tests, predictions)
        ],
    }


def dataset_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split, frame in data.items():
        summary[f"{split}_rows"] = int(len(frame))
        summary[f"{split}_support_coverage"] = float(frame["support"].str.len().gt(0).mean())
    summary["total_rows"] = int(sum(len(frame) for frame in data.values()))
    return summary


def build_project(
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
) -> Tuple[Dict[str, pd.DataFrame], CandidateAnswerModel, ScienceKnowledgeBase, AIStudyAssistantAgent, Dict[str, Any]]:
    data = load_sciq_data(data_dir)

    answer_model = CandidateAnswerModel().fit(data["train"])
    knowledge_base = ScienceKnowledgeBase().fit(data["train"])
    agent = AIStudyAssistantAgent(knowledge_base, data["train"])

    metrics = dataset_summary(data)
    metrics.update(answer_model.evaluate(data["test"]))
    metrics.update(evaluate_intent_router(agent))

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    answer_model.save(model_path / "candidate_answer_model.joblib")
    joblib.dump(knowledge_base, model_path / "science_knowledge_base.joblib")
    with open(model_path / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return data, answer_model, knowledge_base, agent, metrics


def launch_gradio(agent: AIStudyAssistantAgent, share: bool = True) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Run: pip install gradio") from exc

    def chat(message: str, history: List[Any]) -> str:
        del history
        return agent.respond(message)

    demo = gr.ChatInterface(
        fn=chat,
        title="AI Study Assistant Agent",
        description=(
            "Ask a science question, request an explanation, start a practice quiz, "
            "or ask for a study plan. The knowledge comes from the SciQ Kaggle dataset."
        ),
        examples=[
            "What is photosynthesis?",
            "Explain kinetic energy",
            "Quiz me on biology",
            "Make a study plan for chemistry",
        ],
    )
    return demo.launch(share=share)


def run_command_line(agent: AIStudyAssistantAgent) -> None:
    print("AI Study Assistant Agent. Type help for commands and quit to stop.")
    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"quit", "exit"}:
            print("Assistant: Good luck studying.")
            break
        print(f"\nAssistant: {agent.respond(message)}")


if __name__ == "__main__":
    data, answer_model, knowledge_base, agent, metrics = build_project()
    print(json.dumps(metrics, indent=2))
    run_command_line(agent)
