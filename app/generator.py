from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass
class LessonRequest:
    subject: str
    grade_band: str
    topic: str
    class_profile: str
    source_material: str
    accessibility_need: str
    language: str


class LessonGenerator:
    def build_lesson_pack(self, request: LessonRequest) -> dict:
        topic = request.topic.strip() or "the lesson topic"
        subject = request.subject.strip() or "General Studies"
        language = request.language.strip() or "English"

        beginner = dedent(
            f"""
            {topic} in simple terms:
            - Start with one everyday example from home or school.
            - Explain only the main idea in short sentences.
            - Ask students to repeat the concept in their own words.
            - Use one drawing or labeled diagram before introducing definitions.
            """
        ).strip()

        standard = dedent(
            f"""
            Standard classroom explanation for {topic}:
            - Define the core idea clearly.
            - Connect it to the {subject} lesson objective.
            - Walk through one worked example.
            - Ask two comprehension checks during the explanation.
            """
        ).strip()

        advanced = dedent(
            f"""
            Extension path for fast learners:
            - Compare {topic} with a related concept.
            - Introduce one 'why does this happen?' question.
            - Give a challenge prompt that requires reasoning, not recall.
            - Ask students to teach the concept back to a peer.
            """
        ).strip()

        quiz = [
            f"What is the main idea behind {topic}?",
            f"Give one real-life example of {topic}.",
            f"What mistake might someone make when learning {topic} for the first time?",
            f"How would you explain {topic} to a younger student?",
            f"What is one question you still have about {topic}?",
        ]

        game = dedent(
            f"""
            Game: 'Teach, Trade, Fix'
            - Split the class into pairs.
            - Student A explains one part of {topic}.
            - Student B finds one missing step or confusion point.
            - Both pairs trade answers with another pair and improve them.
            """
        ).strip()

        misconceptions = [
            f"Students may memorize {topic} without understanding the underlying process.",
            "Students may confuse definition-based recall with real understanding.",
            "Students may struggle when the same idea is shown in text versus a diagram.",
        ]

        accessibility = dedent(
            f"""
            Accessibility adaptation for {request.accessibility_need or 'mixed learning needs'}:
            - Use shorter instructions and one task at a time.
            - Pair text with visual cues or icons.
            - Offer oral explanation and peer support.
            - Keep the final response format flexible: speak, point, draw, or write.
            """
        ).strip()

        teacher_summary = dedent(
            f"""
            Teacher summary:
            - Best use case: one mixed-ability {request.grade_band} classroom
            - Delivery language: {language}
            - Class profile focus: {request.class_profile or 'mixed pace learners'}
            - Source material used: {'provided' if request.source_material.strip() else 'not provided'}
            """
        ).strip()

        return {
            "beginner": beginner,
            "standard": standard,
            "advanced": advanced,
            "quiz": quiz,
            "game": game,
            "misconceptions": misconceptions,
            "accessibility": accessibility,
            "teacher_summary": teacher_summary,
        }

    def summarize_student_doubts(self, topic: str, raw_answers: str) -> list[str]:
        if not raw_answers.strip():
            return [
                "No student answers provided yet.",
                "Use this section after a quiz or activity to cluster recurring doubts.",
            ]

        return [
            f"Several students seem unsure about the core meaning of {topic}.",
            "Some answers show partial recall but weak reasoning.",
            "At least one group would benefit from a visual reteach and one worked example.",
        ]

    def build_quiz_questions(self, topic: str, concept_names: list[str], language: str) -> list[dict]:
        concept_line = ", ".join(concept_names) if concept_names else topic
        return [
            {
                "question_text": f"Which statement best explains {topic}?",
                "question_type": "mcq",
                "options": {
                    "A": f"It is the main idea of {topic} explained correctly.",
                    "B": f"It means students only memorize {topic}.",
                    "C": f"It is unrelated to {topic}.",
                    "D": f"It is only a diagram and not a concept.",
                },
                "difficulty": "easy",
                "bloom_level": "remember",
                "marks": 2,
                "correct_answer": "A",
                "explanation": f"Use simple {language} wording and focus on the main idea of {topic}.",
            },
            {
                "question_text": f"Explain one key process involved in {topic}.",
                "question_type": "short_answer",
                "options": {},
                "difficulty": "medium",
                "bloom_level": "understand",
                "marks": 4,
                "correct_answer": f"The response should explain one core concept from {concept_line}.",
                "explanation": "Reward reasoning and not just memorized words.",
            },
            {
                "question_text": f"What is one common misunderstanding students may have about {topic}?",
                "question_type": "short_answer",
                "options": {},
                "difficulty": "medium",
                "bloom_level": "analyze",
                "marks": 4,
                "correct_answer": f"A valid answer identifies a misconception related to {topic}.",
                "explanation": "This question is meant to surface conceptual confusion early.",
            },
        ]


DEFAULT_LESSON_GENERATOR = LessonGenerator()


def build_lesson_pack(request: LessonRequest) -> dict:
    return DEFAULT_LESSON_GENERATOR.build_lesson_pack(request)


def summarize_student_doubts(topic: str, raw_answers: str) -> list[str]:
    return DEFAULT_LESSON_GENERATOR.summarize_student_doubts(topic, raw_answers)


def build_quiz_questions(topic: str, concept_names: list[str], language: str) -> list[dict]:
    return DEFAULT_LESSON_GENERATOR.build_quiz_questions(topic, concept_names, language)
