import type { Metadata } from "next";

import { UI_PREFERENCE_BOOTSTRAP_SCRIPT } from "@/lib/ui-preference-bootstrap";

import "./globals.css";

export const metadata: Metadata = {
  title: "Sculpt Workflow Console",
  description: "Visual workflow console for Blender Sculpt agents",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{ __html: UI_PREFERENCE_BOOTSTRAP_SCRIPT }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
