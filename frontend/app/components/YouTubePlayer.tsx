"use client";
import React from "react";

interface YouTubePlayerProps {
    url: string;
    isActive: boolean;
}

export default function YouTubePlayer({ url, isActive }: YouTubePlayerProps) {
    if (!url) return null;

    // Extract video ID
    const videoIdMatch = url.match(/(?:v=|\/)([0-9A-Za-z_-]{11})/);
    const videoId = videoIdMatch ? videoIdMatch[1] : null;

    if (!videoId) {
        return (
            <div className="camera-error">
                <span>Invalid YouTube URL</span>
            </div>
        );
    }

    const embedUrl = `https://www.youtube.com/embed/${videoId}?autoplay=${isActive ? 1 : 0}&mute=0&controls=1&rel=0&modestbranding=1`;

    return (
        <div className="youtube-player-container" style={{ position: "relative" }}>
            <iframe
                width="100%"
                height="100%"
                src={embedUrl}
                title="YouTube lecture video"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                className="youtube-iframe"
            />
            {/* Advice overlay for restricted videos */}
            <div style={{
                position: "absolute",
                bottom: 10,
                left: 10,
                right: 10,
                background: "rgba(0,0,0,0.7)",
                padding: "8px 12px",
                borderRadius: "6px",
                fontSize: "11px",
                color: "#ccc",
                pointerEvents: "none",
                display: isActive ? "block" : "none"
            }}>
                Note: If the video says "Unavailable", the owner has disabled embedding. Notes and snapshots will still be captured on the backend.
            </div>
        </div>
    );
}
