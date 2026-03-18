"use client";
import { useEffect, useMemo } from "react";
import type { ConceptGraph as ConceptGraphType } from "../types";

interface ConceptGraphProps {
    graph: ConceptGraphType | null;
    onFetch: () => void;
    isActive: boolean;
}

// Simple deterministic layout for concept graph nodes
function layoutNodes(nodes: string[]): Record<string, { x: number; y: number }> {
    const positions: Record<string, { x: number; y: number }> = {};
    const cols = 3;
    const xSpacing = 160;
    const ySpacing = 52;
    const padding = 16;

    nodes.forEach((node, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        positions[node] = {
            x: padding + col * xSpacing + (row % 2 === 1 ? 40 : 0),
            y: padding + row * ySpacing,
        };
    });
    return positions;
}

export default function ConceptGraph({ graph, onFetch, isActive }: ConceptGraphProps) {
    useEffect(() => {
        if (isActive) {
            const timer = setInterval(onFetch, 15000);
            onFetch();
            return () => clearInterval(timer);
        }
    }, [isActive, onFetch]);

    const positions = useMemo(() => {
        if (!graph) return {};
        return layoutNodes(graph.nodes);
    }, [graph]);

    const containerHeight = graph ? Math.max(220, Math.ceil(graph.nodes.length / 3) * 52 + 50) : 220;

    return (
        <div className="concept-graph">
            <div className="graph-container" style={{ height: containerHeight }}>
                {!graph ? (
                    <div className="empty-state" style={{ height: "100%" }}>
                        <span className="empty-icon">🕸️</span>
                        <span className="empty-text">Concept graph builds as the lecture progresses</span>
                    </div>
                ) : (
                    <>
                        {/* Edges */}
                        {graph.edges.map((edge, i) => {
                            const from = positions[edge.source];
                            const to = positions[edge.target];
                            if (!from || !to) return null;
                            const dx = to.x - from.x;
                            const dy = to.y - from.y;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            const angle = Math.atan2(dy, dx) * (180 / Math.PI);
                            return (
                                <div
                                    key={`edge-${i}`}
                                    className="graph-edge"
                                    style={{
                                        left: from.x + 40,
                                        top: from.y + 14,
                                        width: len,
                                        transform: `rotate(${angle}deg)`,
                                    }}
                                />
                            );
                        })}
                        {/* Nodes */}
                        {graph.nodes.map((node) => {
                            const pos = positions[node];
                            if (!pos) return null;
                            const isHighlighted = graph.edges.some(
                                (e) => e.source === node || e.target === node
                            );
                            return (
                                <div
                                    key={node}
                                    className={`graph-node ${isHighlighted ? "highlighted" : ""}`}
                                    style={{ left: pos.x, top: pos.y }}
                                    title={node}
                                >
                                    {node}
                                </div>
                            );
                        })}
                    </>
                )}
            </div>
        </div>
    );
}
