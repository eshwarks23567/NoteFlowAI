"use client";
import type { AlertEvent } from "../types";

interface AlertsPanelProps {
    alerts: AlertEvent[];
}

export default function AlertsPanel({ alerts }: AlertsPanelProps) {
    const iconMap: Record<string, string> = {
        critical: "!!",
        warning: "!",
        info: "i",
    };

    return (
        <div className="panel glass-card" style={{ flex: "1 1 auto" }}>
            <div className="panel-header">
                <span className="panel-title">
                    <span className="panel-title-icon">ALT</span>
                    Alerts
                </span>
                {alerts.length > 0 && (
                    <span className="count-badge critical">{alerts.length}</span>
                )}
            </div>
            <div className="panel-body">
                {alerts.length === 0 ? (
                    <div className="empty-state">
                        <span className="empty-icon"></span>
                        <span className="empty-text">Smart alerts will appear when high-importance moments are detected</span>
                    </div>
                ) : (
                    alerts.map((alert) => (
                        <div key={alert.id} className={`alert-item ${alert.alert_type}`}>
                            <span className="alert-icon">{iconMap[alert.alert_type] || "i"}</span>
                            <div className="alert-content">
                                <div className="alert-message">{alert.message}</div>
                                <div className="alert-time">{alert.lecture_time}</div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
