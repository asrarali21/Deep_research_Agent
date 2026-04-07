"use client";

import { Check, Copy } from "lucide-react";
import { type ComponentPropsWithoutRef, useState } from "react";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

import { copyToClipboard } from "@/features/research/utils/copy";

function CodeBlock({ className, children }: ComponentPropsWithoutRef<"code">) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || "");
  const code = String(children).replace(/\n$/, "");

  if (!match) {
    return <code className={className}>{children}</code>;
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={async () => {
          await copyToClipboard(code);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        }}
        className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full border border-white/10 bg-black/20 px-2 py-1 text-xs text-muted transition hover:text-text"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        {copied ? "Copied" : "Copy"}
      </button>
      <SyntaxHighlighter language={match[1]} style={oneDark} customStyle={{ margin: 0, borderRadius: 16 }}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export const markdownComponents = {
  a: ({ href, children, ...props }: ComponentPropsWithoutRef<"a">) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="text-accent underline decoration-accent/30 underline-offset-4 transition hover:decoration-accent"
      {...props}
    >
      {children}
    </a>
  ),
  code: CodeBlock,
};
