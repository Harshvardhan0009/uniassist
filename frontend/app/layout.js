import "./globals.css";

export const metadata = {
  title: "LPUAssist — Your University AI Assistant",
  description:
    "Ask questions about university policies, placement guidelines, dress code, library rules, and more. Powered by RAG technology.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0a0e1a" />
      </head>
      <body>{children}</body>
    </html>
  );
}
