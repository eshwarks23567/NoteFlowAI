import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title: "NoteFlow AI — Live Lecture Note-Taker",
  description: "Real-time AI-powered lecture note-taking system. Captures audio, slides, and behavior to generate structured, importance-ranked notes securely.",
  openGraph: {
    title: "NoteFlow AI — Live Lecture Note-Taker",
    description: "Real-time AI-powered lecture note-taking system.",
    type: "website",
    images: [{
      url: "/og-image.jpg",
      width: 1200,
      height: 630,
      alt: "NoteFlow AI Interface",
    }],
  },
  twitter: {
    card: "summary_large_image",
    title: "NoteFlow AI",
    description: "Real-time AI-powered lecture note-taking system.",
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%230f1115'/><text x='50' y='70' font-family='sans-serif' font-size='60' font-weight='bold' fill='%23e2e4e9' text-anchor='middle'>N</text></svg>" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
