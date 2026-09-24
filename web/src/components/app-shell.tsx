"use client";

import { Menu } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { ChatView } from "./chat-view";
import { LibraryDialog } from "./library-dialog";
import { Sidebar } from "./sidebar";
import { TrustView } from "./trust-view";

function sessionFromPath(pathname: string): string | null {
  const match = /^\/c\/([0-9a-f]{32})\/?$/.exec(pathname);
  return match ? match[1] : null;
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const sessionId = sessionFromPath(pathname);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [chatKey, setChatKey] = useState(0);

  useEffect(() => {
    // Close the mobile drawer after navigating.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMobileOpen(false);
  }, [pathname]);

  const startNewChat = () => {
    window.history.pushState(null, "", "/");
    setChatKey((value) => value + 1);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "o") {
        event.preventDefault();
        startNewChat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex h-full overflow-hidden">
      <div
        className={`fixed inset-0 z-30 bg-black/40 transition-opacity md:hidden ${
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={() => setMobileOpen(false)}
      />
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-72 transition-transform md:static md:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          activeId={sessionId}
          onNewChat={startNewChat}
          onOpenLibrary={() => setLibraryOpen(true)}
        />
      </aside>
      <main className="relative flex min-w-0 flex-1 flex-col">
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          className="absolute left-3 top-3 z-20 rounded-lg p-2 text-zinc-600 hover:bg-zinc-100 md:hidden dark:text-zinc-300 dark:hover:bg-zinc-800"
          aria-label="Open sidebar"
        >
          <Menu className="size-5" />
        </button>
        {pathname === "/trust" ? (
          <TrustView />
        ) : (
          <ChatView key={sessionId ?? `new-${chatKey}`} sessionId={sessionId} onOpenLibrary={() => setLibraryOpen(true)} />
        )}
        {children}
      </main>
      <LibraryDialog open={libraryOpen} onClose={() => setLibraryOpen(false)} />
    </div>
  );
}
