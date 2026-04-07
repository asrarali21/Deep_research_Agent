declare module "react-syntax-highlighter" {
  import type { ComponentType, ReactNode } from "react";

  export type SyntaxHighlighterProps = {
    language?: string;
    style?: Record<string, unknown>;
    customStyle?: Record<string, unknown>;
    children?: ReactNode;
  };

  export const Prism: ComponentType<SyntaxHighlighterProps>;
}

declare module "react-syntax-highlighter/dist/esm/styles/prism" {
  export const oneDark: Record<string, unknown>;
}
