import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Samvaad — Indian Sign Language Communication Aid",
  description:
    "Real-time Indian Sign Language recognition that builds spoken and written transcripts. Designed for deaf and mute communication.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#050816",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/icon.png" />
        <link rel="shortcut icon" href="/icon.png" />
      </head>
      <body>{children}</body>
    </html>
  );
}
