from __future__ import annotations

from app.db import get_connection


def ensure_demo_data() -> None:
    with get_connection() as connection:
        existing_teacher = connection.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
        if existing_teacher:
            connection.execute(
                """
                UPDATE students
                SET email = CASE roll_number
                    WHEN '07A01' THEN 'asha.verma@student.demo'
                    WHEN '07A02' THEN 'rohan.gupta@student.demo'
                    WHEN '07A03' THEN 'meera.khan@student.demo'
                    WHEN '07A04' THEN 'ishaan.rao@student.demo'
                    WHEN '07B01' THEN 'kavya.singh@student.demo'
                    WHEN '07B02' THEN 'dev.malhotra@student.demo'
                    ELSE email
                END
                WHERE email IS NULL OR email = ''
                """
            )
            _ensure_demo_blueprints(connection)
            connection.commit()
            return

        school_id = connection.execute(
            """
            INSERT INTO schools (name, board_type, state, district)
            VALUES (?, ?, ?, ?)
            """,
            ("Sarvodaya Public School", "CBSE", "Delhi", "South West Delhi"),
        ).lastrowid

        teacher_id = connection.execute(
            """
            INSERT INTO teachers (school_id, full_name, email, google_account_email)
            VALUES (?, ?, ?, ?)
            """,
            (school_id, "Aditi Sharma", "aditi.sharma@example.edu", "aditi.sharma@gmail.com"),
        ).lastrowid

        science_class_id = connection.execute(
            """
            INSERT INTO classes (school_id, teacher_id, academic_year, grade, section, subject, medium)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (school_id, teacher_id, "2026-27", "7", "A", "Science", "English"),
        ).lastrowid

        math_class_id = connection.execute(
            """
            INSERT INTO classes (school_id, teacher_id, academic_year, grade, section, subject, medium)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (school_id, teacher_id, "2026-27", "7", "B", "Math", "English + Hindi"),
        ).lastrowid

        students = [
            (school_id, science_class_id, "07A01", "Asha Verma", "asha.verma@student.demo", "Hindi", "Needs visual reinforcement"),
            (school_id, science_class_id, "07A02", "Rohan Gupta", "rohan.gupta@student.demo", "English", ""),
            (school_id, science_class_id, "07A03", "Meera Khan", "meera.khan@student.demo", "English + Hindi", "Slow processing speed"),
            (school_id, science_class_id, "07A04", "Ishaan Rao", "ishaan.rao@student.demo", "English", ""),
            (school_id, math_class_id, "07B01", "Kavya Singh", "kavya.singh@student.demo", "Hindi", ""),
            (school_id, math_class_id, "07B02", "Dev Malhotra", "dev.malhotra@student.demo", "English", ""),
        ]
        connection.executemany(
            """
            INSERT INTO students (
                school_id, class_id, roll_number, full_name, email, preferred_language, accessibility_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            students,
        )

        chapter_id = connection.execute(
            """
            INSERT INTO chapters (board_type, grade, subject, chapter_code, chapter_name, term)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("CBSE", "7", "Science", "SCI-CH-01", "Nutrition in Plants", "Term 1"),
        ).lastrowid

        concepts = [
            (chapter_id, "PHOTOSYNTHESIS", "Photosynthesis", "How plants make food", "core"),
            (chapter_id, "STOMATA", "Role of Stomata", "Gas exchange in leaves", "medium"),
            (chapter_id, "NUTRIENTS", "Nutrient Sources", "Difference between sunlight, soil, and water roles", "core"),
        ]
        connection.executemany(
            """
            INSERT INTO concepts (chapter_id, concept_code, concept_name, description, difficulty_level)
            VALUES (?, ?, ?, ?, ?)
            """,
            concepts,
        )

        assessment_id = connection.execute(
            """
            INSERT INTO assessments (
                class_id, chapter_id, teacher_id, title, assessment_type, delivery_mode,
                google_form_id, google_form_url, language, total_marks, assigned_at, due_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                science_class_id,
                chapter_id,
                teacher_id,
                "Nutrition in Plants Checkpoint Quiz",
                "class_test",
                "google_form",
                "demo-form-001",
                "https://docs.google.com/forms/d/demo-form-001/edit",
                "English + Hindi",
                10,
                "2026-05-06 09:00:00",
                "2026-05-06 15:00:00",
            ),
        ).lastrowid

        questions = [
            (
                assessment_id,
                1,
                "What do plants use to make their own food?",
                "mcq",
                "easy",
                "remember",
                2,
                "Sunlight, water, and carbon dioxide",
                "Plants need sunlight, water, and carbon dioxide for photosynthesis.",
                "gq-001",
            ),
            (
                assessment_id,
                2,
                "Why are stomata important for leaves?",
                "short_answer",
                "medium",
                "understand",
                4,
                "They help in gas exchange.",
                "Stomata allow carbon dioxide in and oxygen out.",
                "gq-002",
            ),
            (
                assessment_id,
                3,
                "Does soil act as food for plants? Explain briefly.",
                "short_answer",
                "medium",
                "analyze",
                4,
                "No. Soil provides minerals and water, but plants make food themselves.",
                "Plants do not eat soil. Soil supplies water and minerals.",
                "gq-003",
            ),
        ]
        connection.executemany(
            """
            INSERT INTO assessment_questions (
                assessment_id, question_number, question_text, question_type, difficulty,
                bloom_level, marks, correct_answer, explanation, google_question_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            questions,
        )

        question_rows = connection.execute(
            "SELECT id, question_number FROM assessment_questions WHERE assessment_id = ? ORDER BY question_number",
            (assessment_id,),
        ).fetchall()
        concept_rows = connection.execute(
            "SELECT id, concept_code FROM concepts WHERE chapter_id = ?",
            (chapter_id,),
        ).fetchall()
        concept_by_code = {row["concept_code"]: row["id"] for row in concept_rows}

        question_links = [
            (question_rows[0]["id"], concept_by_code["PHOTOSYNTHESIS"], 0.7),
            (question_rows[0]["id"], concept_by_code["NUTRIENTS"], 0.3),
            (question_rows[1]["id"], concept_by_code["STOMATA"], 1.0),
            (question_rows[2]["id"], concept_by_code["NUTRIENTS"], 1.0),
        ]
        connection.executemany(
            """
            INSERT INTO question_concepts (assessment_question_id, concept_id, weightage)
            VALUES (?, ?, ?)
            """,
            question_links,
        )

        science_students = connection.execute(
            "SELECT id, full_name FROM students WHERE class_id = ? ORDER BY roll_number",
            (science_class_id,),
        ).fetchall()

        student_assessments = [
            (assessment_id, science_students[0]["id"], "graded", 7, 70, "2026-05-06 11:20:00", "2026-05-06 12:00:00"),
            (assessment_id, science_students[1]["id"], "graded", 9, 90, "2026-05-06 11:05:00", "2026-05-06 12:00:00"),
            (assessment_id, science_students[2]["id"], "graded", 4, 40, "2026-05-06 11:40:00", "2026-05-06 12:00:00"),
            (assessment_id, science_students[3]["id"], "graded", 6, 60, "2026-05-06 11:10:00", "2026-05-06 12:00:00"),
        ]
        connection.executemany(
            """
            INSERT INTO student_assessments (
                assessment_id, student_id, status, score_obtained, percentage, submitted_at, graded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            student_assessments,
        )

        attempt_rows = connection.execute(
            "SELECT id, student_id FROM student_assessments WHERE assessment_id = ?",
            (assessment_id,),
        ).fetchall()
        attempt_by_student = {row["student_id"]: row["id"] for row in attempt_rows}

        answers = [
            (attempt_by_student[science_students[0]["id"]], question_rows[0]["id"], "Sunlight, water and air", "sunlight, water and carbon dioxide", 1, 2, "Good recall.", None, "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[0]["id"]], question_rows[1]["id"], "They help plants breathe.", "gas exchange", 1, 3, "Partially correct, but mention carbon dioxide and oxygen.", "incomplete_reasoning", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[0]["id"]], question_rows[2]["id"], "Soil is the food of the plant.", "soil is food", 0, 2, "Confused nutrients with food production.", "concept_misunderstanding", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[1]["id"]], question_rows[0]["id"], "Sunlight, water, and carbon dioxide", "sunlight, water, and carbon dioxide", 1, 2, "Correct.", None, "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[1]["id"]], question_rows[1]["id"], "Stomata are tiny openings for gas exchange.", "gas exchange", 1, 4, "Correct.", None, "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[1]["id"]], question_rows[2]["id"], "No. Plants make food by photosynthesis and soil gives minerals.", "plants make food, soil gives minerals", 1, 3, "Strong answer.", None, "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[2]["id"]], question_rows[0]["id"], "Plants use soil and water.", "soil and water", 0, 0, "Missed sunlight and carbon dioxide.", "concept_misunderstanding", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[2]["id"]], question_rows[1]["id"], "I do not know.", "unknown", 0, 1, "Needs reteach with diagram.", "incomplete_reasoning", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[2]["id"]], question_rows[2]["id"], "Yes, soil is food.", "soil is food", 0, 3, "Strong misconception around nutrient sources.", "concept_misunderstanding", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[3]["id"]], question_rows[0]["id"], "Sunlight, water, and carbon dioxide", "sunlight, water, and carbon dioxide", 1, 2, "Correct.", None, "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[3]["id"]], question_rows[1]["id"], "For air exchange", "air exchange", 1, 2, "Basic understanding present.", "incomplete_reasoning", "2026-05-06 12:00:00"),
            (attempt_by_student[science_students[3]["id"]], question_rows[2]["id"], "Soil gives food to roots.", "soil gives food", 0, 2, "Needs concept correction.", "concept_misunderstanding", "2026-05-06 12:00:00"),
        ]
        connection.executemany(
            """
            INSERT INTO student_answers (
                student_assessment_id, assessment_question_id, raw_answer, normalized_answer,
                is_correct, score_awarded, feedback, error_type, processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            answers,
        )

        mastery_rows = [
            (science_students[0]["id"], concept_by_code["PHOTOSYNTHESIS"], science_class_id, 0.72, 0.76, 2, 2, "2026-05-06 12:00:00", "developing"),
            (science_students[0]["id"], concept_by_code["STOMATA"], science_class_id, 0.68, 0.62, 1, 1, "2026-05-06 12:00:00", "developing"),
            (science_students[0]["id"], concept_by_code["NUTRIENTS"], science_class_id, 0.35, 0.70, 2, 0, "2026-05-06 12:00:00", "lagging"),
            (science_students[1]["id"], concept_by_code["PHOTOSYNTHESIS"], science_class_id, 0.94, 0.90, 2, 2, "2026-05-06 12:00:00", "strong"),
            (science_students[1]["id"], concept_by_code["STOMATA"], science_class_id, 0.92, 0.84, 1, 1, "2026-05-06 12:00:00", "strong"),
            (science_students[1]["id"], concept_by_code["NUTRIENTS"], science_class_id, 0.88, 0.82, 2, 1, "2026-05-06 12:00:00", "strong"),
            (science_students[2]["id"], concept_by_code["PHOTOSYNTHESIS"], science_class_id, 0.22, 0.78, 2, 0, "2026-05-06 12:00:00", "lagging"),
            (science_students[2]["id"], concept_by_code["STOMATA"], science_class_id, 0.30, 0.66, 1, 0, "2026-05-06 12:00:00", "lagging"),
            (science_students[2]["id"], concept_by_code["NUTRIENTS"], science_class_id, 0.18, 0.82, 2, 0, "2026-05-06 12:00:00", "lagging"),
            (science_students[3]["id"], concept_by_code["PHOTOSYNTHESIS"], science_class_id, 0.75, 0.72, 2, 2, "2026-05-06 12:00:00", "developing"),
            (science_students[3]["id"], concept_by_code["STOMATA"], science_class_id, 0.58, 0.60, 1, 1, "2026-05-06 12:00:00", "developing"),
            (science_students[3]["id"], concept_by_code["NUTRIENTS"], science_class_id, 0.40, 0.74, 2, 0, "2026-05-06 12:00:00", "lagging"),
        ]
        connection.executemany(
            """
            INSERT INTO student_concept_mastery (
                student_id, concept_id, class_id, mastery_score, confidence_score,
                questions_attempted, questions_correct, last_assessed_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            mastery_rows,
        )

        class_mastery_rows = [
            (science_class_id, concept_by_code["PHOTOSYNTHESIS"], 0.66, 4, 1, "2026-05-06 12:00:00"),
            (science_class_id, concept_by_code["STOMATA"], 0.62, 4, 1, "2026-05-06 12:00:00"),
            (science_class_id, concept_by_code["NUTRIENTS"], 0.45, 4, 3, "2026-05-06 12:00:00"),
        ]
        connection.executemany(
            """
            INSERT INTO class_concept_mastery (
                class_id, concept_id, average_mastery_score, students_assessed, students_lagging, last_updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            class_mastery_rows,
        )

        remediation_rows = [
            (science_students[0]["id"], assessment_id, concept_by_code["NUTRIENTS"], "worksheet", "Use a soil-vs-food comparison worksheet with diagrams.", 3, "mock-gemma"),
            (science_students[2]["id"], assessment_id, concept_by_code["PHOTOSYNTHESIS"], "reteach", "Reteach photosynthesis with a picture sequence and Hindi support.", 5, "mock-gemma"),
            (science_students[2]["id"], assessment_id, concept_by_code["NUTRIENTS"], "peer_learning", "Pair with Rohan for concept explanation before the next quiz.", 4, "mock-gemma"),
            (science_students[3]["id"], assessment_id, concept_by_code["NUTRIENTS"], "practice_quiz", "Assign a 3-question remedial mini-quiz on nutrient sources.", 4, "mock-gemma"),
        ]
        connection.executemany(
            """
            INSERT INTO remediation_recommendations (
                student_id, assessment_id, concept_id, recommendation_type,
                recommendation_text, priority, generated_by_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            remediation_rows,
        )

        sync_rows = [
            (assessment_id, "form_create", "demo-form-001", "success", "Demo Google Form created."),
            (assessment_id, "response_fetch", "demo-sync-001", "success", "Fetched four graded responses."),
        ]
        connection.executemany(
            """
            INSERT INTO google_sync_logs (assessment_id, sync_type, external_id, status, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            sync_rows,
        )

        blueprint_rows = [
            (
                science_students[0]["id"],
                science_class_id,
                '["Can identify the main purpose of photosynthesis", "Understands basic plant parts involved in food making"]',
                '["Confuses the role of soil with the process of food production", "Struggles to explain nutrient sources precisely in science answers"]',
                '["Can strengthen concept clarity through chapter-wise remedial quizzes", "Can improve written science explanations with diagram-based prompts"]',
                '["If nutrient-source misconceptions continue, later plant nutrition chapters may become harder", "Weak descriptive science answers may reduce test scores even when recall is present"]',
                '["Revise the difference between food production and mineral absorption", "Practice 3 short science answers on plant nutrition each week"]',
                "Asha shows working understanding of the basic idea of photosynthesis, but she still confuses how plants make food with what they absorb from soil. Her next improvement area is concept precision in plant nutrition topics.",
                "mock-gemma-blueprint",
                1,
                "2026-05-06 11:20:00",
            ),
            (
                science_students[1]["id"],
                science_class_id,
                '["Strong understanding of photosynthesis inputs and outputs", "Understands the role of stomata and leaf function accurately"]',
                '["Needs more higher-order challenge questions beyond direct textbook recall"]',
                '["Can move into comparative biology questions and deeper reasoning", "Can reinforce learning by solving extension quizzes on plant processes"]',
                '["If only basic questions are given, conceptual growth may plateau"]',
                '["Assign advanced application questions on plant nutrition", "Give compare-and-explain tasks involving stomata and gas exchange"]',
                "Rohan has strong chapter mastery in Nutrition in Plants and is ready for advanced conceptual practice. He benefits more from extension questions than from repeated basic recall checks.",
                "mock-gemma-blueprint",
                1,
                "2026-05-06 11:05:00",
            ),
            (
                science_students[2]["id"],
                science_class_id,
                '["Can recall that sunlight is related to plant food making", "Shows early understanding when science content is broken into steps"]',
                '["Does not yet understand the full photosynthesis process", "Confuses oxygen release and food production", "Finds it difficult to connect textbook explanation with diagram meaning"]',
                '["Can improve through stepwise reteaching and repeated concept quizzes", "Visual and bilingual question practice can improve science chapter retention"]',
                '["Foundational gaps in plant nutrition can affect later biology units", "Weak chapter confidence may reduce willingness to attempt written science answers"]',
                '["Re-teach photosynthesis as a sequence of inputs, process, and outputs", "Use one concept-focused remedial quiz after each reteach session", "Practice labeling-based science questions"]',
                "Meera currently has major concept gaps in Nutrition in Plants, especially around the process of photosynthesis and the roles of gases and soil. Her blueprint should focus on foundational science correction before moving ahead.",
                "mock-gemma-blueprint",
                1,
                "2026-05-06 11:40:00",
            ),
            (
                science_students[3]["id"],
                science_class_id,
                '["Can answer direct recall questions on plant food making", "Recognizes that leaves are central to photosynthesis"]',
                '["Needs stronger understanding of why plant processes happen", "Still mixes surface recall with full chapter understanding"]',
                '["Can improve by solving reasoning-based science questions", "Can strengthen chapter mastery through misconception-focused mini quizzes"]',
                '["If only recall is practiced, deeper chapter understanding may remain weak"]',
                '["Add explanation-based questions after every recall question", "Use quizzes that ask why sunlight, water, and carbon dioxide are all necessary"]',
                "Ishaan has basic recall of the chapter, but his science understanding remains shallow in reasoning-based questions. His blueprint should emphasize moving from textbook recall to concept explanation.",
                "mock-gemma-blueprint",
                1,
                "2026-05-06 11:10:00",
            ),
        ]
        connection.executemany(
            """
            INSERT INTO student_blueprints (
                student_id, class_id, subject, strengths_json, weaknesses_json, opportunities_json,
                threats_json, recommendations_json, narrative, generated_by_model,
                based_on_assessments, last_submission_at
            )
            VALUES (?, ?, 'Science', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint_rows,
        )

        connection.commit()


def _ensure_demo_blueprints(connection) -> None:
    student_rows = connection.execute(
        """
        SELECT s.id, s.roll_number, s.class_id
        FROM students s
        JOIN classes c ON c.id = s.class_id
        WHERE c.subject = 'Science' AND c.grade = '7' AND c.section = 'A'
        ORDER BY s.roll_number
        """
    ).fetchall()
    if not student_rows:
        return

    class_id = student_rows[0]["class_id"]
    by_roll = {row["roll_number"]: row["id"] for row in student_rows}
    blueprint_rows = [
        (
            by_roll["07A01"],
            class_id,
            '["Can identify the main purpose of photosynthesis", "Understands basic plant parts involved in food making"]',
            '["Confuses the role of soil with the process of food production", "Struggles to explain nutrient sources precisely in science answers"]',
            '["Can strengthen concept clarity through chapter-wise remedial quizzes", "Can improve written science explanations with diagram-based prompts"]',
            '["If nutrient-source misconceptions continue, later plant nutrition chapters may become harder", "Weak descriptive science answers may reduce test scores even when recall is present"]',
            '["Revise the difference between food production and mineral absorption", "Practice 3 short science answers on plant nutrition each week"]',
            "Asha shows working understanding of the basic idea of photosynthesis, but she still confuses how plants make food with what they absorb from soil. Her next improvement area is concept precision in plant nutrition topics.",
            "mock-gemma-blueprint",
            1,
            "2026-05-06 11:20:00",
        ),
        (
            by_roll["07A02"],
            class_id,
            '["Strong understanding of photosynthesis inputs and outputs", "Understands the role of stomata and leaf function accurately"]',
            '["Needs more higher-order challenge questions beyond direct textbook recall"]',
            '["Can move into comparative biology questions and deeper reasoning", "Can reinforce learning by solving extension quizzes on plant processes"]',
            '["If only basic questions are given, conceptual growth may plateau"]',
            '["Assign advanced application questions on plant nutrition", "Give compare-and-explain tasks involving stomata and gas exchange"]',
            "Rohan has strong chapter mastery in Nutrition in Plants and is ready for advanced conceptual practice. He benefits more from extension questions than from repeated basic recall checks.",
            "mock-gemma-blueprint",
            1,
            "2026-05-06 11:05:00",
        ),
        (
            by_roll["07A03"],
            class_id,
            '["Can recall that sunlight is related to plant food making", "Shows early understanding when science content is broken into steps"]',
            '["Does not yet understand the full photosynthesis process", "Confuses oxygen release and food production", "Finds it difficult to connect textbook explanation with diagram meaning"]',
            '["Can improve through stepwise reteaching and repeated concept quizzes", "Visual and bilingual question practice can improve science chapter retention"]',
            '["Foundational gaps in plant nutrition can affect later biology units", "Weak chapter confidence may reduce willingness to attempt written science answers"]',
            '["Re-teach photosynthesis as a sequence of inputs, process, and outputs", "Use one concept-focused remedial quiz after each reteach session", "Practice labeling-based science questions"]',
            "Meera currently has major concept gaps in Nutrition in Plants, especially around the process of photosynthesis and the roles of gases and soil. Her blueprint should focus on foundational science correction before moving ahead.",
            "mock-gemma-blueprint",
            1,
            "2026-05-06 11:40:00",
        ),
        (
            by_roll["07A04"],
            class_id,
            '["Can answer direct recall questions on plant food making", "Recognizes that leaves are central to photosynthesis"]',
            '["Needs stronger understanding of why plant processes happen", "Still mixes surface recall with full chapter understanding"]',
            '["Can improve by solving reasoning-based science questions", "Can strengthen chapter mastery through misconception-focused mini quizzes"]',
            '["If only recall is practiced, deeper chapter understanding may remain weak"]',
            '["Add explanation-based questions after every recall question", "Use quizzes that ask why sunlight, water, and carbon dioxide are all necessary"]',
            "Ishaan has basic recall of the chapter, but his science understanding remains shallow in reasoning-based questions. His blueprint should emphasize moving from textbook recall to concept explanation.",
            "mock-gemma-blueprint",
            1,
            "2026-05-06 11:10:00",
        ),
    ]
    connection.executemany(
        """
        INSERT INTO student_blueprints (
            student_id, class_id, subject, strengths_json, weaknesses_json, opportunities_json,
            threats_json, recommendations_json, narrative, generated_by_model,
            based_on_assessments, last_submission_at
        )
        VALUES (?, ?, 'Science', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id, class_id) DO UPDATE SET
            strengths_json = excluded.strengths_json,
            weaknesses_json = excluded.weaknesses_json,
            opportunities_json = excluded.opportunities_json,
            threats_json = excluded.threats_json,
            recommendations_json = excluded.recommendations_json,
            narrative = excluded.narrative,
            generated_by_model = excluded.generated_by_model,
            based_on_assessments = excluded.based_on_assessments,
            last_submission_at = excluded.last_submission_at,
            updated_at = CURRENT_TIMESTAMP
        """,
        blueprint_rows,
    )
