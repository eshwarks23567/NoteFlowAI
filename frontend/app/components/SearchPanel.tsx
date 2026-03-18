"use client";
import { useState } from "react";

interface SearchPanelProps {
    onSearch: (query: string) => Promise<any[]>;
}

export default function SearchPanel({ onSearch }: SearchPanelProps) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<any[]>([]);
    const [searching, setSearching] = useState(false);

    const handleSearch = async () => {
        if (!query.trim()) return;
        setSearching(true);
        const res = await onSearch(query.trim());
        setResults(res);
        setSearching(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter") handleSearch();
    };

    return (
        <div className="panel glass-card" style={{ flex: "0 0 auto" }}>
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">🔍</span>
                    Search
                </span>
            </div>
            <div className="panel-body">
                <div className="search-input-wrapper">
                    <span className="search-icon">🔎</span>
                    <input
                        type="text"
                        className="search-input"
                        placeholder="When did professor mention..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                </div>
                {searching && (
                    <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: 8 }}>
                        Searching...
                    </div>
                )}
                {results.map((r, i) => (
                    <div key={i} className="search-result">
                        <div className="search-result-time">
                            [{r.lecture_time}] — {r.type === "concept" ? "💡 Concept" : "📝 Transcript"}
                        </div>
                        <div className="search-result-text">
                            {r.title || r.text}
                        </div>
                    </div>
                ))}
                {!searching && results.length === 0 && query && (
                    <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: 16 }}>
                        No results found
                    </div>
                )}
            </div>
        </div>
    );
}
