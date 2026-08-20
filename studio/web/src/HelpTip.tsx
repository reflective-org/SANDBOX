// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The hover explanation: a CircleHelp icon that reveals the field's description and provenance.
 *
 * Requested in review: boxes should not carry walls of explanation text -- a question mark you can
 * hover. The pattern (and the icon) come from the SAI simulator's ScenarioBuilder, which pairs
 * every control with a zero-delay tooltip.
 *
 * A <button>, not a styled span: keyboard users reach it with Tab and the tip opens on focus, so
 * hover is a convenience rather than the only door. No positioning library -- the tip anchors to
 * the icon and flips are not needed at tooltip sizes on a scrolling page.
 */

import { CircleHelp } from "lucide-react";

export function HelpTip({
  children,
  label = "explanation",
}: {
  children: React.ReactNode;
  label?: string;
}) {
  return (
    <span className="helptip">
      <button type="button" className="helptip-trigger" aria-label={label}>
        <CircleHelp size={14} strokeWidth={2} />
      </button>
      <span className="helptip-body" role="tooltip">
        {children}
      </span>
    </span>
  );
}
