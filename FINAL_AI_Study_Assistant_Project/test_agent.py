import pandas as pd

from ai_study_assistant import (
    AIStudyAssistantAgent,
    CandidateAnswerModel,
    ScienceKnowledgeBase,
    clean_sciq_frame,
    evaluate_intent_router,
)


def make_small_test_data() -> pd.DataFrame:
    concepts = [
        ("photosynthesis", "plants use sunlight to make food", "respiration", "erosion", "freezing"),
        ("kinetic energy", "the energy of motion", "potential energy", "heat", "voltage"),
        ("gravity", "a force that pulls masses together", "friction", "refraction", "diffusion"),
        ("cell membrane", "controls what enters and leaves a cell", "nucleus", "ribosome", "chloroplast"),
        ("voltage", "the common word for electric potential difference", "current", "resistance", "power"),
    ]
    rows = []
    for index in range(100):
        answer, meaning, first, second, third = concepts[index % len(concepts)]
        rows.append(
            {
                "question": f"Question {index}: What term means {meaning}?",
                "correct_answer": answer,
                "distractor1": first,
                "distractor2": second,
                "distractor3": third,
                "support": f"{answer.title()} is {meaning}.",
            }
        )
    return clean_sciq_frame(pd.DataFrame(rows))


def test_project_components() -> None:
    frame = make_small_test_data()
    train = frame.iloc[:75]
    test = frame.iloc[75:]

    answer_model = CandidateAnswerModel(max_features=3000).fit(train)
    metrics = answer_model.evaluate(test)
    assert 0.0 <= metrics["multiple_choice_accuracy"] <= 1.0

    knowledge_base = ScienceKnowledgeBase(max_features=3000).fit(train)
    results = knowledge_base.search("What is photosynthesis?", top_k=1)
    assert results
    assert results[0].answer == "photosynthesis"

    agent = AIStudyAssistantAgent(knowledge_base, train)
    intent_metrics = evaluate_intent_router(agent)
    assert intent_metrics["intent_accuracy"] == 1.0
    assert "photosynthesis" in agent.respond("What is photosynthesis?").lower()
    assert "practice question" in agent.respond("quiz me on biology").lower()


if __name__ == "__main__":
    test_project_components()
    print("All local component tests passed.")
