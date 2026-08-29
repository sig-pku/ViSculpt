"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

type Props = {
  children: ReactNode;
};

function subscribeToDocumentBody() {
  return () => undefined;
}

function getDocumentBody(): HTMLElement | null {
  return typeof document === "undefined" ? null : document.body;
}

function getServerDocumentBody(): null {
  return null;
}

export function ModalPortal({ children }: Props) {
  const portalTarget = useSyncExternalStore(
    subscribeToDocumentBody,
    getDocumentBody,
    getServerDocumentBody,
  );

  return portalTarget ? createPortal(children, portalTarget) : null;
}
