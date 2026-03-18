import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NoteFlow AI — Live Lecture Note-Taker",
  description: "Real-time AI-powered lecture note-taking with multi-modal analysis. Captures audio, slides, and professor behavior to generate structured, importance-ranked notes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>🎓</text></svg>" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
