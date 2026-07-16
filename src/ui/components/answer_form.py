"""Answer form components for supported question types."""

from __future__ import annotations

from uuid import UUID

import streamlit as st

from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion
from src.models.study_session import ConfidenceLevel, FillBlankLearnerAnswer, LearnerAnswer, MCQLearnerAnswer
from src.ui.helpers import build_widget_key


CONFIDENCE_LABELS = {
    "en": {
        ConfidenceLevel.GUESSED: "1 — Guessed",
        ConfidenceLevel.LOW: "2 — Low confidence",
        ConfidenceLevel.MEDIUM: "3 — Moderately confident",
        ConfidenceLevel.HIGH: "4 — Confident",
        ConfidenceLevel.VERY_HIGH: "5 — Very confident",
    },
    "tr": {
        ConfidenceLevel.GUESSED: "1 — Tahmin ettim",
        ConfidenceLevel.LOW: "2 — Güvenim düşük",
        ConfidenceLevel.MEDIUM: "3 — Orta düzeyde eminim",
        ConfidenceLevel.HIGH: "4 — Eminim",
        ConfidenceLevel.VERY_HIGH: "5 — Çok eminim",
    },
}


def render_answer_form(
    question: GeneratedQuestion,
    *,
    session_id: UUID,
    confidence_required: bool,
    language: str = "English",
) -> tuple[bool, LearnerAnswer | None, ConfidenceLevel | None]:
    language_code = _language_code(language)
    confidence_labels = CONFIDENCE_LABELS[language_code]
    unknown_label = "Bilmiyorum" if language_code == "tr" else "I do not know"
    form_key = build_widget_key(
        "answer-form",
        session_id=session_id,
        question_id=question.id,
        position=question.position,
    )
    with st.form(key=form_key, clear_on_submit=False):
        unknown = st.checkbox(
            unknown_label,
            help=(
                "Tahmin etmek yerine açıklamayı görmek için bunu seçin."
                if language_code == "tr"
                else "Choose this to see the explanation without guessing."
            ),
            key=build_widget_key(
                "answer-unknown",
                session_id=session_id,
                question_id=question.id,
                position=question.position,
            ),
        )
        learner_answer: LearnerAnswer | None = None
        if isinstance(question, MCQQuestion):
            option_text = {option.id: option.text for option in question.options}
            selected = st.radio(
                "Choose one answer",
                list(option_text),
                index=None,
                disabled=unknown,
                format_func=lambda option_id: f"{option_id}. {option_text[option_id]}",
                key=build_widget_key(
                    "answer-mcq-option",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
            if unknown:
                learner_answer = MCQLearnerAnswer(unknown=True)
            elif selected:
                learner_answer = MCQLearnerAnswer(selected_option_id=selected)
        elif isinstance(question, FillBlankQuestion):
            submitted = st.text_input(
                "Your answer",
                placeholder="Type the missing word or phrase",
                disabled=unknown,
                key=build_widget_key(
                    "answer-fill-blank",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
            if unknown:
                learner_answer = FillBlankLearnerAnswer(unknown=True)
            elif submitted.strip():
                learner_answer = FillBlankLearnerAnswer(submitted_answer=submitted)

        confidence_value = None
        if confidence_required:
            confidence_value = st.selectbox(
                "Ne kadar eminsiniz?" if language_code == "tr" else "How confident are you?",
                options=[None, *confidence_labels],
                format_func=lambda value: (
                    ("Güven düzeyi seçin" if language_code == "tr" else "Select confidence")
                    if value is None
                    else confidence_labels[value]
                ),
                help=(
                    "Güven düzeyi tekrar planlamasına yardımcı olur; doğruluğu değiştirmez."
                    if language_code == "tr"
                    else "Confidence helps schedule useful reviews. It never changes correctness."
                ),
                key=build_widget_key(
                    "answer-confidence",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
        submitted_form = st.form_submit_button(
            "Check answer",
            type="primary",
            icon=":material/check:",
            width="stretch",
        )

    if submitted_form and learner_answer is None:
        st.error(
            "Bir yanıt seçin veya “Bilmiyorum” seçeneğini işaretleyin."
            if language_code == "tr"
            else "Choose an answer or select “I do not know.”"
        )
        return False, None, None
    if submitted_form and confidence_required and confidence_value is None:
        st.error(
            "Yanıtınızı kontrol etmeden önce bir güven düzeyi seçin."
            if language_code == "tr"
            else "Choose a confidence level before checking your answer."
        )
        return False, None, None
    return submitted_form, learner_answer, confidence_value


def _language_code(language: str) -> str:
    normalized = language.strip().casefold()
    return "tr" if normalized.startswith("tr") or normalized in {"turkish", "türkçe", "turkce"} else "en"
