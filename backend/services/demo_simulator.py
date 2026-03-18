"""Demo Simulator — generates realistic lecture events for frontend testing."""
from __future__ import annotations
import asyncio, random, time, json
from typing import Callable, Awaitable
from models.schemas import (
    TranscriptionEvent, SlideEvent, GestureEvent, ImportanceEvent,
    KeyConcept, SummaryUpdate, AlertEvent, ConceptLink,
    ImportanceLevel, SpeakerRole, GestureType, EventType, WSMessage
)
from services.fusion_engine import fuse_scores, compute_keyword_score


# ── Sample lecture data (Machine Learning Basics) ────────────────

SLIDES = [
    {
        "number": 1, "title": "Machine Learning Basics — Week 3",
        "content": "Topics: Gradient Descent, Loss Functions, Overfitting vs Underfitting, Regularization",
        "has_diagram": False, "has_equation": False,
    },
    {
        "number": 2, "title": "What is Gradient Descent?",
        "content": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent. Learning rate (α) controls step size.",
        "has_diagram": True, "has_equation": True,
    },
    {
        "number": 3, "title": "Loss Functions",
        "content": "MSE = (1/n) Σ(yᵢ - ŷᵢ)². Cross-entropy loss for classification. The choice of loss function affects convergence speed and model behavior.",
        "has_diagram": False, "has_equation": True,
    },
    {
        "number": 4, "title": "Learning Rate Selection",
        "content": "Too high → divergence. Too low → slow convergence. Adaptive methods: Adam, RMSProp, AdaGrad. Learning rate scheduling.",
        "has_diagram": True, "has_equation": False,
    },
    {
        "number": 5, "title": "Overfitting vs Underfitting",
        "content": "Overfitting: model memorizes training data, poor generalization. Underfitting: model too simple, can't capture patterns. Bias-variance tradeoff.",
        "has_diagram": True, "has_equation": False,
    },
    {
        "number": 6, "title": "Regularization Techniques",
        "content": "L1 (Lasso): feature selection via sparsity. L2 (Ridge): weight decay. Dropout: randomly deactivate neurons during training. Early stopping.",
        "has_diagram": False, "has_equation": True,
    },
    {
        "number": 7, "title": "Practical Tips & Summary",
        "content": "Start with simple models. Use cross-validation. Monitor training/validation curves. Ensemble methods for better performance.",
        "has_diagram": False, "has_equation": False,
    },
]

TRANSCRIPT_SEGMENTS = [
    # (speaker, text, has_emphasis, keywords, gesture, gesture_intensity, voice_emphasis)
    (SpeakerRole.PROFESSOR, "Welcome back everyone. Today we're continuing with machine learning fundamentals, specifically gradient descent.", False, ["gradient descent"], GestureType.NONE, 0.1, 0.2),
    (SpeakerRole.PROFESSOR, "Let's start by understanding what gradient descent actually does. It's an optimization algorithm.", False, ["optimization"], GestureType.POINTING, 0.3, 0.3),
    (SpeakerRole.PROFESSOR, "This is the backbone of training neural networks. Remember that. Without gradient descent, we cannot train deep learning models.", True, ["gradient descent", "neural networks"], GestureType.EMPHASIS, 0.9, 0.85),
    (SpeakerRole.PROFESSOR, "Think of it like rolling a ball down a hill. The ball naturally rolls toward the lowest point — that's the minimum of our loss function.", False, ["loss function"], GestureType.SWEEPING, 0.6, 0.5),
    (SpeakerRole.PROFESSOR, "The learning rate alpha controls how big each step is. Too large and you overshoot. Too small and it takes forever.", False, ["learning rate"], GestureType.COUNTING, 0.5, 0.6),
    (SpeakerRole.STUDENT, "Professor, how do we know which learning rate to pick?", False, ["learning rate"], GestureType.NONE, 0.0, 0.3),
    (SpeakerRole.PROFESSOR, "Great question! In practice, we use adaptive learning rate methods like Adam or RMSProp. They automatically adjust the learning rate during training.", False, ["Adam", "RMSProp"], GestureType.COUNTING, 0.4, 0.5),
    (SpeakerRole.PROFESSOR, "Now let's talk about loss functions. Pay attention to this — the choice of loss function fundamentally changes how your model learns.", True, ["loss functions"], GestureType.EMPHASIS, 0.85, 0.9),
    (SpeakerRole.PROFESSOR, "For regression problems, we typically use Mean Squared Error. For classification, cross-entropy loss is the standard choice.", False, ["MSE", "cross-entropy"], GestureType.POINTING, 0.4, 0.4),
    (SpeakerRole.PROFESSOR, "Here's a crucial point. The most common mistake beginners make is confusing overfitting with underfitting.", True, ["overfitting", "underfitting"], GestureType.EMPHASIS, 0.95, 0.92),
    (SpeakerRole.PROFESSOR, "Overfitting means your model has memorized the training data. It performs great on training set but terrible on new data.", False, ["overfitting"], GestureType.SWEEPING, 0.6, 0.55),
    (SpeakerRole.STUDENT, "Is that like when the training accuracy is 99% but test accuracy is 60%?", False, ["accuracy"], GestureType.NONE, 0.0, 0.3),
    (SpeakerRole.PROFESSOR, "Exactly! That's a textbook example of overfitting. And this is why we always split our data into training, validation, and test sets.", True, ["overfitting", "validation"], GestureType.EMPHASIS, 0.7, 0.75),
    (SpeakerRole.PROFESSOR, "Now, underfitting is the opposite — your model is too simple to capture the underlying patterns in the data.", False, ["underfitting"], GestureType.SWEEPING, 0.5, 0.5),
    (SpeakerRole.PROFESSOR, "The key to solving overfitting is regularization. L1 regularization, also called Lasso, promotes sparsity in your weights.", False, ["regularization", "L1", "Lasso"], GestureType.POINTING, 0.5, 0.55),
    (SpeakerRole.PROFESSOR, "L2 regularization, or Ridge, penalizes large weights but doesn't zero them out. And then there's dropout — don't forget about dropout.", True, ["L2", "Ridge", "dropout"], GestureType.COUNTING, 0.65, 0.7),
    (SpeakerRole.PROFESSOR, "Dropout is essential for deep networks. During training, we randomly deactivate neurons. This prevents the network from relying too heavily on any single neuron.", True, ["dropout"], GestureType.EMPHASIS, 0.8, 0.8),
    (SpeakerRole.STUDENT, "Does dropout change during testing?", False, ["dropout"], GestureType.NONE, 0.0, 0.2),
    (SpeakerRole.PROFESSOR, "Yes! During testing, all neurons are active, but we scale the outputs. This is a very important distinction that many people miss.", True, ["dropout", "testing"], GestureType.HANDS_RAISED, 0.75, 0.8),
    (SpeakerRole.PROFESSOR, "Let me summarize the key takeaways. Gradient descent is your core optimizer. Choose loss functions wisely. Watch for overfitting. Use regularization.", True, ["gradient descent", "loss functions", "overfitting", "regularization"], GestureType.COUNTING, 0.7, 0.75),
]

KEY_CONCEPTS_DATA = [
    {
        "title": "Gradient Descent",
        "definition": "Optimization algorithm to minimize the loss function by iteratively moving in the direction of steepest descent",
        "professor_quote": "This is the backbone of training neural networks",
        "gesture_note": "Professor used strong emphasis gestures while explaining",
        "related": ["Loss Functions", "Learning Rate", "Neural Networks"],
        "importance": 0.88,
        "slide": 2,
        "sources": ["slide", "gesture", "voice"],
    },
    {
        "title": "Loss Functions",
        "definition": "Mathematical functions that measure how well a model's predictions match the actual data",
        "professor_quote": "The choice of loss function fundamentally changes how your model learns",
        "gesture_note": "Pointed at equation on slide repeatedly",
        "related": ["MSE", "Cross-Entropy", "Gradient Descent"],
        "importance": 0.78,
        "slide": 3,
        "sources": ["slide", "voice"],
    },
    {
        "title": "Overfitting vs Underfitting",
        "definition": "Overfitting: model memorizes training data. Underfitting: model too simple to capture patterns. The bias-variance tradeoff.",
        "professor_quote": "Most common mistake beginners make",
        "gesture_note": "Drew curve in air 3 times to illustrate",
        "related": ["Regularization", "Validation", "Bias-Variance"],
        "importance": 0.92,
        "slide": 5,
        "sources": ["slide", "gesture", "voice"],
    },
    {
        "title": "Regularization",
        "definition": "Techniques to prevent overfitting: L1 (Lasso), L2 (Ridge), Dropout, Early Stopping",
        "professor_quote": "Don't forget about dropout — it's essential for deep networks",
        "gesture_note": "Counted techniques on fingers",
        "related": ["Overfitting", "Dropout", "L1", "L2"],
        "importance": 0.75,
        "slide": 6,
        "sources": ["slide", "gesture", "voice"],
    },
    {
        "title": "Learning Rate",
        "definition": "Hyperparameter controlling step size in gradient descent. Too high → divergence. Too low → slow convergence.",
        "professor_quote": "In practice, we use adaptive learning rate methods like Adam",
        "gesture_note": "",
        "related": ["Gradient Descent", "Adam", "RMSProp"],
        "importance": 0.62,
        "slide": 4,
        "sources": ["slide", "voice"],
    },
]

CONCEPT_GRAPH_NODES = [
    "Gradient Descent", "Loss Functions", "Learning Rate", "MSE", "Cross-Entropy",
    "Overfitting", "Underfitting", "Regularization", "L1 / Lasso", "L2 / Ridge",
    "Dropout", "Neural Networks", "Adam", "Bias-Variance Tradeoff"
]

CONCEPT_GRAPH_EDGES = [
    ("Gradient Descent", "Loss Functions", "minimizes"),
    ("Gradient Descent", "Learning Rate", "controlled_by"),
    ("Gradient Descent", "Neural Networks", "trains"),
    ("Loss Functions", "MSE", "includes"),
    ("Loss Functions", "Cross-Entropy", "includes"),
    ("Learning Rate", "Adam", "adaptive_method"),
    ("Overfitting", "Underfitting", "opposite_of"),
    ("Overfitting", "Regularization", "solved_by"),
    ("Regularization", "L1 / Lasso", "includes"),
    ("Regularization", "L2 / Ridge", "includes"),
    ("Regularization", "Dropout", "includes"),
    ("Overfitting", "Bias-Variance Tradeoff", "relates_to"),
]


class DemoSimulator:
    """Generates realistic lecture events on a timer."""

    def __init__(self, send_callback: Callable[[WSMessage], Awaitable[None]]):
        self.send = send_callback
        self.running = False
        self.start_time = 0.0
        self.current_slide_idx = 0
        self.transcript_idx = 0
        self.concept_idx = 0
        self.summary_counter = 0

    def _lecture_time(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    async def start(self):
        self.running = True
        self.start_time = time.time()
        # Send session started
        await self.send(WSMessage(
            event_type=EventType.SESSION_STATUS,
            data={"status": "started", "title": "Machine Learning Basics — Week 3"}
        ))
        # Send first slide
        await self._send_slide(0)
        # Start event loops
        asyncio.create_task(self._transcript_loop())
        asyncio.create_task(self._slide_loop())
        asyncio.create_task(self._summary_loop())

    async def stop(self):
        self.running = False
        await self.send(WSMessage(
            event_type=EventType.SESSION_STATUS,
            data={"status": "stopped"}
        ))

    async def _send_slide(self, idx: int):
        if idx >= len(SLIDES):
            return
        slide_data = SLIDES[idx]
        event = SlideEvent(
            slide_number=slide_data["number"],
            title=slide_data["title"],
            content_text=slide_data["content"],
            has_diagram=slide_data["has_diagram"],
            has_equation=slide_data["has_equation"],
            lecture_time=self._lecture_time(),
        )
        await self.send(WSMessage(
            event_type=EventType.SLIDE_CHANGE,
            data=event.model_dump()
        ))

    async def _transcript_loop(self):
        """Emit transcript segments at realistic intervals."""
        while self.running and self.transcript_idx < len(TRANSCRIPT_SEGMENTS):
            await asyncio.sleep(random.uniform(3.5, 7.0))
            if not self.running:
                break

            seg = TRANSCRIPT_SEGMENTS[self.transcript_idx]
            speaker, text, has_emphasis, keywords, gesture, g_intensity, v_emphasis = seg

            # Send transcription event
            t_event = TranscriptionEvent(
                text=text,
                speaker=speaker,
                confidence=round(random.uniform(0.85, 0.98), 2),
                is_emphasis_phrase=has_emphasis,
                keywords=keywords,
                lecture_time=self._lecture_time(),
            )
            await self.send(WSMessage(
                event_type=EventType.TRANSCRIPTION,
                data=t_event.model_dump()
            ))

            # Send gesture event if applicable
            if gesture != GestureType.NONE:
                g_event = GestureEvent(
                    gesture_type=gesture,
                    intensity=g_intensity,
                    duration=round(random.uniform(1.0, 4.0), 1),
                    description=f"Professor: {gesture.value} gesture detected",
                    lecture_time=self._lecture_time(),
                )
                await self.send(WSMessage(
                    event_type=EventType.GESTURE,
                    data=g_event.model_dump()
                ))

            # Send importance event for emphasized segments
            if has_emphasis or g_intensity > 0.6:
                kw_score = compute_keyword_score(text)
                imp = fuse_scores(
                    gesture_intensity=g_intensity,
                    voice_emphasis=v_emphasis,
                    slide_dwell_time=random.uniform(0.3, 0.7),
                    keyword_score=kw_score,
                    question_score=0.2 if speaker == SpeakerRole.STUDENT else 0.0,
                    lecture_time=self._lecture_time(),
                    trigger_text=text[:80],
                )
                await self.send(WSMessage(
                    event_type=EventType.IMPORTANCE,
                    data=imp.model_dump()
                ))

            # Send key concept events at computed intervals
            if has_emphasis and self.concept_idx < len(KEY_CONCEPTS_DATA):
                concept_data = KEY_CONCEPTS_DATA[self.concept_idx]
                kc = KeyConcept(
                    title=concept_data["title"],
                    definition=concept_data["definition"],
                    importance_score=concept_data["importance"],
                    importance_level=(
                        ImportanceLevel.CRITICAL if concept_data["importance"] > 0.7
                        else ImportanceLevel.IMPORTANT if concept_data["importance"] >= 0.4
                        else ImportanceLevel.SUPPORTING
                    ),
                    professor_quote=concept_data["professor_quote"],
                    gesture_note=concept_data["gesture_note"],
                    related_concepts=concept_data["related"],
                    slide_number=concept_data["slide"],
                    sources=concept_data["sources"],
                    lecture_time=self._lecture_time(),
                )
                await self.send(WSMessage(
                    event_type=EventType.KEY_CONCEPT,
                    data=kc.model_dump()
                ))
                self.concept_idx += 1

            # Send occasional alerts
            if has_emphasis and random.random() > 0.5:
                alert = AlertEvent(
                    message=f"⚡ High-importance moment: {keywords[0] if keywords else 'concept'} — pay attention!",
                    alert_type="critical" if g_intensity > 0.8 else "warning",
                    lecture_time=self._lecture_time(),
                )
                await self.send(WSMessage(
                    event_type=EventType.ALERT,
                    data=alert.model_dump()
                ))

            self.transcript_idx += 1

        # End of demo
        if self.running:
            await asyncio.sleep(3)
            await self.stop()

    async def _slide_loop(self):
        """Advance slides periodically."""
        self.current_slide_idx = 0
        while self.running:
            await asyncio.sleep(random.uniform(18, 30))
            if not self.running:
                break
            self.current_slide_idx += 1
            if self.current_slide_idx < len(SLIDES):
                await self._send_slide(self.current_slide_idx)

    async def _summary_loop(self):
        """Send summary updates every ~30 seconds."""
        summaries = [
            ("Introduction to Gradient Descent", [
                "Gradient descent is the core optimization algorithm for ML",
                "It minimizes the loss function iteratively",
                "Learning rate controls the step size",
            ]),
            ("Loss Functions & Selection", [
                "MSE used for regression, cross-entropy for classification",
                "Loss function choice affects convergence speed",
                "Professor emphasized this is fundamental to model training",
            ]),
            ("Overfitting vs Underfitting", [
                "Overfitting: memorizing training data → poor generalization",
                "Underfitting: model too simple to capture patterns",
                "Always split data into train/validation/test sets",
                "🔴 CRITICAL: Professor spent significant time on this topic",
            ]),
            ("Regularization Techniques", [
                "L1 (Lasso): promotes sparse weights, feature selection",
                "L2 (Ridge): penalizes large weights without zeroing",
                "Dropout: randomly deactivates neurons during training",
                "Essential for preventing overfitting in deep networks",
            ]),
            ("Key Takeaways & Review", [
                "Gradient descent → core optimizer",
                "Choose loss functions wisely for your problem type",
                "Watch for overfitting, use regularization",
                "Adaptive methods (Adam) simplify learning rate selection",
            ]),
        ]
        idx = 0
        while self.running and idx < len(summaries):
            await asyncio.sleep(30)
            if not self.running:
                break
            topic, bullets = summaries[idx]
            level = ImportanceLevel.CRITICAL if idx == 2 else ImportanceLevel.IMPORTANT
            su = SummaryUpdate(
                current_topic=topic,
                bullet_points=bullets,
                importance_level=level,
            )
            await self.send(WSMessage(
                event_type=EventType.SUMMARY_UPDATE,
                data=su.model_dump()
            ))
            idx += 1
