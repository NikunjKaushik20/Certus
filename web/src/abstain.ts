/* Why an eye went to a person instead of being decided automatically, in words a technician or an
   ophthalmologist can act on. The keys are Engine.decide's abstain_reason values. */
const REASONS: Record<string, string> = {
  ambiguous: "The score sits inside the screening band, too close to call either way.",
  readers_disagree:
    "The two readers did not agree. Certus and the independent whole-image grader must both make " +
    "the same call before an eye is referred or cleared automatically.",
  low_reliability: "The photograph or the camera is not reliable enough to believe any score.",
  outside_conformal_set: "The score falls outside every calibrated prediction set.",
};

export function abstainText(reason: string | null | undefined): string {
  if (!reason) return "";
  return REASONS[reason] ?? reason;
}
