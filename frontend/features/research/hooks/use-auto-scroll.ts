"use client";

import { RefObject, useEffect, useMemo, useState } from "react";

export function useAutoScroll(containerRef: RefObject<HTMLElement | null>, deps: unknown[]) {
  const [isPinnedToBottom, setIsPinnedToBottom] = useState(true);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }

    const onScroll = () => {
      const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
      setIsPinnedToBottom(distanceFromBottom <= 96);
    };

    onScroll();
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => element.removeEventListener("scroll", onScroll);
  }, [containerRef]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element || !isPinnedToBottom) {
      return;
    }

    element.scrollTo({
      top: element.scrollHeight,
      behavior: "smooth",
    });
  }, [containerRef, isPinnedToBottom, ...deps]);

  return useMemo(
    () => ({
      isPinnedToBottom,
      scrollToBottom: () =>
        containerRef.current?.scrollTo({
          top: containerRef.current.scrollHeight,
          behavior: "smooth",
        }),
    }),
    [containerRef, isPinnedToBottom],
  );
}
