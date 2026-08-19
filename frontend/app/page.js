"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Image from "next/image";
import styles from "./page.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const SUGGESTIONS = [
  "What is the dress code policy?",
  "Tell me about placement policies",
  "What are the library rules?",
  "What residential facilities are available?",
  "How does semester exchange work?",
  "What academic benefits are offered?",
];

export default function Home() {
  const [chatSessions, setChatSessions] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Load chat sessions from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("lpuassist_chats");
      if (saved) {
        setChatSessions(JSON.parse(saved));
      }
    } catch (e) {
      console.error("Failed to load chat history:", e);
    }
  }, []);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem("lpuassist_chats", JSON.stringify(chatSessions));
    } catch (e) {
      console.error("Failed to save chat history:", e);
    }
  }, [chatSessions]);

  // Persist active chat messages into sessions list
  useEffect(() => {
    if (activeChatId && messages.length > 0) {
      setChatSessions((prev) => {
        const idx = prev.findIndex((s) => s.id === activeChatId);
        const updated = {
          id: activeChatId,
          title: messages[0]?.content?.slice(0, 40) || "New chat",
          messages,
          updatedAt: Date.now(),
        };
        if (idx >= 0) {
          const copy = [...prev];
          copy[idx] = updated;
          return copy;
        }
        return [updated, ...prev];
      });
    }
  }, [messages, activeChatId]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  }, [input]);

  const sendMessage = async (text) => {
    const question = text || input.trim();
    if (!question || isLoading) return;

    // Create a new session if none is active
    if (!activeChatId) {
      setActiveChatId(Date.now().toString());
    }

    setMessages((prev) => [...prev, { role: "user", content: question, id: Date.now() }]);
    setInput("");
    setIsLoading(true);

    // Prior turns (exclude errors) give the backend follow-up context.
    const history = messages
      .filter((m) => !m.isError)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-8);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      const data = await res.json();

      setMessages((prev) => [...prev, {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        id: Date.now() + 1,
      }]);
    } catch (err) {
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: `Something went wrong: ${err.message}. Please make sure the backend server is running.`,
        isError: true,
        id: Date.now() + 1,
      }]);
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setInput("");
  };

  const handleLoadChat = (session) => {
    setActiveChatId(session.id);
    setMessages(session.messages);
    setInput("");
  };

  const handleDeleteChat = (e, sessionId) => {
    e.stopPropagation();
    setChatSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeChatId === sessionId) {
      setActiveChatId(null);
      setMessages([]);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className={styles.app}>
      {/* ── Sidebar ──────────────────────────────────────────── */}
      <aside className={styles.sidebar}>
        <button className={styles.newChatBtn} onClick={handleNewChat}>
          <PlusIcon /> New chat
        </button>

        <div className={styles.sidebarNav}>
          <div className={styles.navSection}>
            <span className={styles.navLabel}>Recent</span>
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={`${styles.navItem} ${session.id === activeChatId ? styles.navItemActive : ""}`}
                onClick={() => handleLoadChat(session)}
              >
                <ChatBubbleIcon />
                <span className={styles.navText}>
                  {session.title}
                </span>
                <button
                  className={styles.deleteBtn}
                  onClick={(e) => handleDeleteChat(e, session.id)}
                  title="Delete chat"
                >
                  <TrashIcon />
                </button>
              </div>
            ))}
            {chatSessions.length === 0 && (
              <div className={styles.navEmpty}>No conversations yet</div>
            )}
          </div>
        </div>

        <div className={styles.sidebarFooter}>
          <div className={styles.footerItem}>
            <Image src="/lpu-logo.webp" alt="LPU" width={20} height={20} />
            <span>LPUAssist v1.0</span>
          </div>
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────────────── */}
      <main className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          <span className={styles.modelName}>LPUAssist</span>
        </header>

        {/* Chat Area */}
        <div className={styles.chatArea}>
          {isEmpty ? (
            <div className={styles.empty}>
              <div className={styles.emptyLogo}>
                <Image src="/lpu-logo.webp" alt="LPU Logo" width={64} height={64} priority />
              </div>
              <h1 className={styles.emptyTitle}>What can I help with?</h1>
              <div className={styles.suggestions}>
                {SUGGESTIONS.map((s) => (
                  <button key={s} className={styles.chip} onClick={() => sendMessage(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className={styles.messages}>
              {messages.map((msg) => (
                <div key={msg.id} className={`${styles.msgRow} ${msg.role === "user" ? styles.msgRowUser : ""}`}>
                  {msg.role === "assistant" && (
                    <div className={styles.avatar}>
                      <Image src="/lpu-logo.webp" alt="LPU" width={28} height={28} />
                    </div>
                  )}
                  <div className={`${styles.msgContent} ${msg.role === "user" ? styles.msgContentUser : ""}`}>
                    {msg.role === "assistant" && <div className={styles.msgLabel}>LPUAssist</div>}
                    <div className={`${styles.msgText} ${msg.isError ? styles.msgError : ""}`}>
                      <FormattedText text={msg.content} />
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className={styles.srcRow}>
                        {msg.sources.map((s) => (
                          <span key={s} className={styles.srcBadge}>
                            📄 {s.replace(".pdf", "")}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className={styles.msgRow}>
                  <div className={styles.avatar}>
                    <Image src="/lpu-logo.webp" alt="LPU" width={28} height={28} />
                  </div>
                  <div className={styles.msgContent}>
                    <div className={styles.msgLabel}>LPUAssist</div>
                    <div className={styles.thinking}>
                      <div className={styles.dot} />
                      <div className={styles.dot} />
                      <div className={styles.dot} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className={styles.inputBar}>
          <div className={styles.inputBox}>
            <textarea
              ref={textareaRef}
              className={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask LPUAssist anything..."
              rows={1}
              disabled={isLoading}
            />
            <button
              className={styles.sendBtn}
              onClick={() => sendMessage()}
              disabled={!input.trim() || isLoading}
            >
              {isLoading ? (
                <div className={styles.spinner} />
              ) : (
                <ArrowUpIcon />
              )}
            </button>
          </div>
          <p className={styles.disclaimer}>
            LPUAssist retrieves answers from official university documents. Always verify critical information.
          </p>
        </div>
      </main>
    </div>
  );
}

/* ── Text Formatter ──────────────────────────────────────────────── */
function FormattedText({ text }) {
  if (!text) return null;

  // Safety net: strip any leaked <think>...</think> blocks from reasoning models
  let cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  cleaned = cleaned.replace(/<think>[\s\S]*$/g, "").trim();
  cleaned = cleaned.replace(/<\/?think>/g, "").trim();

  return cleaned.split("\n").map((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) return <br key={i} />;
    if (trimmed === "---") return <hr key={i} className={styles.hr} />;

    // Bold headers: **text**
    if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
      return <h4 key={i} className={styles.h4}>{trimmed.replace(/\*\*/g, "")}</h4>;
    }

    // Bold prefix: **From Source:**
    const boldMatch = trimmed.match(/^\*\*(.*?)\*\*(.*)$/);
    if (boldMatch) {
      return <p key={i} className={styles.p}><strong>{boldMatch[1]}</strong>{boldMatch[2]}</p>;
    }

    // Italic note: *text*
    if (trimmed.startsWith("*") && trimmed.endsWith("*") && !trimmed.startsWith("**")) {
      return <p key={i} className={styles.note}>{trimmed.replace(/^\*|\*$/g, "")}</p>;
    }

    // Bullet points
    if (trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
      return <li key={i} className={styles.li}>{trimmed.slice(2)}</li>;
    }

    return <p key={i} className={styles.p}>{trimmed}</p>;
  });
}

/* ── Icons ────────────────────────────────────────────────────────── */
function PlusIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>;
}
function ChatBubbleIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}
function TrashIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}
function ArrowUpIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 19V5M5 12l7-7 7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}
