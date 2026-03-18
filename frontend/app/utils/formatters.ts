/* ── Utility functions ── */

export function formatLectureTime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function importanceStars(score: number): string {
    if (score > 0.85) return "⭐⭐⭐";
    if (score > 0.7) return "⭐⭐";
    if (score > 0.5) return "⭐";
    return "";
}

export function importanceColor(level: string): string {
    switch (level) {
        case "critical": return "var(--color-critical)";
        case "important": return "var(--color-important)";
        default: return "var(--color-supporting)";
    }
}

export function importanceBadge(level: string): string {
    switch (level) {
        case "critical": return "🔴";
        case "important": return "🟡";
        default: return "🟢";
    }
}

export function gestureIcon(type: string): string {
    switch (type) {
        case "pointing": return "👉";
        case "sweeping": return "👋";
        case "counting": return "🤚";
        case "emphasis": return "✊";
        case "hands_raised": return "🙌";
        case "leaning_forward": return "🧍";
        default: return "🤲";
    }
}

export function speakerIcon(role: string): string {
    return role === "professor" ? "👨‍🏫" : role === "student" ? "🙋" : "❓";
}

export function truncate(text: string, max: number): string {
    return text.length > max ? text.slice(0, max) + "…" : text;
}
