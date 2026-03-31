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
        <div className="youtube-player-container">
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
        </div>
    );
}
