"use client";
import type { SlideEvent } from "../types";

interface SlideViewerProps {
    currentSlide: SlideEvent | null;
}

export default function SlideViewer({ currentSlide }: SlideViewerProps) {
    if (!currentSlide) {
        return (
            <div className="slide-viewer">
                <div className="empty-state">
                    <span className="empty-icon">SLIDE</span>
                    <span className="empty-text">Current slide will be displayed here</span>
                </div>
            </div>
        );
    }

    return (
        <div className="slide-viewer">
            <div className="slide-card">
                {currentSlide.snapshot_url && (
                    <div className="slide-image-container">
                        <img 
                            src={currentSlide.snapshot_url} 
                            alt={currentSlide.title} 
                            className="slide-image"
                        />
                    </div>
                )}
                <div className="slide-number">
                    Slide {currentSlide.slide_number} — {currentSlide.lecture_time}
                </div>
                <div className="slide-title">{currentSlide.title}</div>
                <div className="slide-content">{currentSlide.content_text}</div>
                <div className="slide-badges">
                    {currentSlide.has_diagram && <span className="slide-badge">Diagram</span>}
                    {currentSlide.has_equation && <span className="slide-badge">Equation</span>}
                </div>
            </div>
        </div>
    );
}
