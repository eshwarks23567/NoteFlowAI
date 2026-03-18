"use client";
import type { KeyConcept } from "../types";
import { importanceStars, importanceBadge } from "../utils/formatters";

interface KeyConceptsPanelProps {
    concepts: KeyConcept[];
}

export default function KeyConceptsPanel({ concepts }: KeyConceptsPanelProps) {
    return (
        <div className="panel concepts-panel glass-card">
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">💡</span>
                    Key Concepts
                </span>
                {concepts.length > 0 && (
                    <span className={`count-badge ${concepts.some(c => c.importance_level === "critical") ? "critical" : ""}`}>
                        {concepts.length}
                    </span>
                )}
            </div>
            <div className="panel-body">
                {concepts.length === 0 ? (
                    <div className="empty-state">
                        <span className="empty-icon">🔍</span>
                        <span className="empty-text">Key concepts will appear here as they&apos;re detected during the lecture</span>
                    </div>
                ) : (
                    concepts.map((concept) => (
                        <div key={concept.id} className={`concept-card ${concept.importance_level}`}>
                            <div className="concept-header">
                                <span className="concept-title">
                                    {importanceBadge(concept.importance_level)} {concept.title}
                                </span>
                                <span className="concept-time">{concept.lecture_time}</span>
                            </div>
                            <span className="concept-stars">{importanceStars(concept.importance_score)}</span>
                            <div className="concept-definition">{concept.definition}</div>
                            {concept.professor_quote && (
                                <div className="concept-quote">&ldquo;{concept.professor_quote}&rdquo;</div>
                            )}
                            {concept.gesture_note && (
                                <div className="concept-gesture">🤲 {concept.gesture_note}</div>
                            )}
                            {concept.related_concepts.length > 0 && (
                                <div className="concept-tags">
                                    {concept.related_concepts.map((rc, i) => (
                                        <span key={i} className="concept-tag">{rc}</span>
                                    ))}
                                </div>
                            )}
                            {concept.sources.length > 0 && (
                                <div className="concept-sources">
                                    {concept.sources.map((s, i) => (
                                        <span key={i} className="source-badge">{s}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
