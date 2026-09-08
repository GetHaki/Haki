"use client";

import { CheckCircle2, Clock, Coins, CreditCard, Infinity as InfinityIcon, Sparkles, XCircle, Zap } from "lucide-react";
import { useState } from "react";
import { useConsole } from "@/components/console/provider";
import { ErrorState, Loading, PageHeader } from "@/components/console/states";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { HakiApiError } from "@/lib/api";
import { Label } from "@/components/ui/label";
import { AppTranslations, useAppT } from "@/lib/app/translations";
import { useLocale } from "@/lib/i18n/use-locale";
import { useApi } from "@/lib/use-api";
import { BillingStatus, CreditsBalance } from "@/lib/types";

// Sprint 12/13: single plan, no picker yet (documented scope limit — see
// app/api/routes/billing.py). Source of truth for the price is the server
// config (HAKI_BILLING_CLOUD_PLAN_PRICE_XOF) — this label just mirrors it.
type PlanKey = "starter" | "growth" | "scale";
function cloudPlans(t: AppTranslations) {
  return {
    starter: {
      name: "Starter",
      priceLabel: `10 000 FCFA / ${t.blMonth}`,
      creditsLabel: `20 000 ${t.blCreditsIncludedMonthly}`,
    },
    growth: {
      name: "Growth",
      priceLabel: `25 000 FCFA / ${t.blMonth}`,
      creditsLabel: `50 000 ${t.blCreditsIncludedMonthly}`,
    },
    scale: {
      name: "Scale",
      priceLabel: `75 000 FCFA / ${t.blMonth}`,
      creditsLabel: `150 000 ${t.blCreditsIncludedMonthly}`,
    },
  } as const;
}
const HIGHLIGHTED_PLAN: PlanKey = "growth";

// Quick top-up amounts (credits), 0,65 FCFA/crédit — see
// HAKI_BILLING_CREDIT_PRICE_XOF_PER_CREDIT.
const CREDIT_PRICE_XOF = 0.65;
const QUICK_TOPUPS = [1000, 5000, 20000];

async function fetchBillingStatus(): Promise<BillingStatus> {
  const response = await fetch("/api/billing/status", { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new HakiApiError(
      response.status,
      body?.error?.type ?? "unknown_error",
      body?.error?.message ?? `Erreur API (HTTP ${response.status})`,
    );
  }
  return response.json();
}

async function fetchCredits(): Promise<CreditsBalance> {
  const response = await fetch("/api/billing/credits", { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new HakiApiError(
      response.status,
      body?.error?.type ?? "unknown_error",
      body?.error?.message ?? `Erreur API (HTTP ${response.status})`,
    );
  }
  return response.json();
}

function statusLabels(t: AppTranslations): Record<string, string> {
  return {
    active: t.blStatusActive,
    trialing: t.blStatusTrialing,
    past_due: t.blStatusPastDue,
    cancelled: t.blStatusCancelled,
    pending: t.blStatusPending,
  };
}

function reasonLabels(t: AppTranslations): Record<string, string> {
  return {
    signup_grant: t.blReasonSignupGrant,
    monthly_free_grant: t.blReasonMonthlyFreeGrant,
    subscription_grant: t.blReasonSubscriptionGrant,
    topup_purchase: t.blReasonTopupPurchase,
    capture_debit: t.blReasonCaptureDebit,
    admin_adjustment: t.blReasonAdminAdjustment,
  };
}

function StatusBadge({ status }: { status: string }) {
  const t = useAppT();
  const label = statusLabels(t)[status] ?? status;
  const isGood = status === "active" || status === "trialing";
  const isBad = status === "past_due" || status === "cancelled";
  const Icon = isGood ? CheckCircle2 : isBad ? XCircle : Clock;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[10.5px] tracking-widest uppercase ${
        isGood
          ? "border-hkc-ok/30 bg-hkc-ok/10 text-hkc-ok"
          : isBad
            ? "border-hkc-sup/30 bg-hkc-sup/10 text-hkc-sup"
            : "border-hkc-line bg-hkc-hover text-hkc-mut"
      }`}
    >
      <Icon className="size-3" />
      {label}
    </span>
  );
}

function CreditsCard() {
  const t = useAppT();
  const locale = useLocale();
  const dateLocale = locale === "fr" ? "fr-FR" : "en-US";
  const { data, error, loading, reload } = useApi<CreditsBalance>(fetchCredits, []);
  const [amount, setAmount] = useState<number>(QUICK_TOPUPS[0]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // No phone number collected here: the checkout page picks the payment
  // method (mobile money or card) and phone number itself once redirected.
  async function topUp() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const response = await fetch("/api/billing/credits", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credits: amount }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(body?.error?.message ?? `Erreur API (HTTP ${response.status})`);
      }
      if (body?.payment_url) {
        window.location.href = body.payment_url;
        return;
      }
      reload();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : t.ovUnknownError);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="rounded-lg border border-hkc-line bg-hkc-surface px-5 py-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-mono text-[10px] tracking-widest text-hkc-faint uppercase">
            {t.blCreditsBalance}
          </p>
          {!error && loading && <div className="mt-1 h-7 w-24"><Loading rows={1} /></div>}
          {!error && !loading && data && (
            <p className="mt-0.5 flex items-center gap-2 text-lg text-hkc-ink">
              <Coins className="size-4 text-hkc-accent" />
              {data.credit_balance.toLocaleString(dateLocale)} {t.blCreditsWord}
            </p>
          )}
        </div>
      </div>

      <p className="mt-2 text-[13px] text-hkc-mut">{t.blCreditsExplain}</p>

      {error && <ErrorState message={error} onRetry={reload} />}

      {!error && !loading && (
        <>
          <div className="mt-4 border-t border-hkc-line pt-4">
            <Label className="text-hkc-mut">{t.blTopUpLabel}</Label>
            <div className="mt-2 flex flex-wrap gap-2">
              {QUICK_TOPUPS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setAmount(q)}
                  className={`rounded-md border px-3 py-1.5 font-mono text-[12.5px] transition-colors ${
                    amount === q
                      ? "border-hkc-accent bg-hkc-chip text-hkc-accent"
                      : "border-hkc-line bg-hkc-bg text-hkc-mut hover:bg-hkc-hover"
                  }`}
                >
                  {q.toLocaleString(dateLocale)}
                </button>
              ))}
            </div>
            <p className="mt-2 font-mono text-[11.5px] text-hkc-faint">
              {amount.toLocaleString(dateLocale)} {t.blCreditsWord} ={" "}
              {(amount * CREDIT_PRICE_XOF).toLocaleString(dateLocale)} FCFA {t.blPricePerCredit}
            </p>

            {submitError && (
              <p className="mt-3 rounded-md border border-hkc-sup/40 bg-hkc-sup/10 px-3 py-2 text-[13px] text-hkc-sup">
                {submitError}
              </p>
            )}

            <Button
              onClick={topUp}
              disabled={submitting}
              className="mt-4 bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
            >
              <Zap className="size-4" />
              {submitting
                ? t.blRedirecting
                : `${t.blTopUpLabel} ${amount.toLocaleString(dateLocale)} ${t.blCreditsWord}`}
            </Button>
          </div>

          {data && data.transactions.length > 0 && (
            <div className="mt-5 border-t border-hkc-line pt-4">
              <p className="font-mono text-[10px] tracking-widest text-hkc-faint uppercase">
                {t.blHistoryTitle}
              </p>
              <div className="mt-2 space-y-1.5">
                {data.transactions.map((tx) => (
                  <div
                    key={tx.id}
                    className="flex items-center justify-between font-mono text-[12px] text-hkc-mut"
                  >
                    <span>{reasonLabels(t)[tx.reason] ?? tx.reason}</span>
                    <span className={tx.delta >= 0 ? "text-hkc-ok" : "text-hkc-faint"}>
                      {tx.delta >= 0 ? "+" : ""}
                      {tx.delta.toLocaleString(dateLocale)}
                    </span>
                    <span className="text-hkc-faint">
                      {new Date(tx.created_at).toLocaleDateString(dateLocale)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// La facturation par crédits (abonnement Cloud + recharges GeniusPay) ne
// concerne que les comptes provisionnés via Clerk (voir Organization.owner_
// ref côté API) : un compte self-hosted n'a jamais de ligne `organizations`
// et reste gratuit et illimité (app/billing/credits.py). GET /api/billing/*
// exige en plus une session Clerk pour résoudre l'owner_ref — inexistante en
// mode self-hosted — donc appeler ces routes ici renverrait toujours 401 et
// déclencherait la déconnexion automatique sur 401 de useApi (lib/use-api.ts),
// éjectant l'utilisateur self-hosted de la console entière. On évite ces
// appels dans ce mode et on explique la situation à la place.
function SelfHostedNotice({ t }: { t: AppTranslations }) {
  return (
    <div className="rounded-lg border border-hkc-line bg-hkc-surface px-5 py-5">
      <div className="flex items-center gap-2">
        <InfinityIcon className="size-4 text-hkc-accent" />
        <p className="text-lg text-hkc-ink">{t.blSelfHostedTitle}</p>
      </div>
      <p className="mt-2 text-[13px] text-hkc-mut">{t.blSelfHostedBody}</p>
    </div>
  );
}

function PlanCards({
  t,
  onSelect,
  ctaLabel,
}: {
  t: AppTranslations;
  onSelect: (plan: PlanKey) => void;
  ctaLabel: (plan: PlanKey) => string;
}) {
  const plans = cloudPlans(t);
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {(Object.entries(plans) as [PlanKey, (typeof plans)[PlanKey]][]).map(([key, p]) => {
        const highlighted = key === HIGHLIGHTED_PLAN;
        return (
          <div
            key={key}
            className={`relative flex flex-col rounded-lg border px-4 py-5 ${
              highlighted ? "border-hkc-accent bg-hkc-chip" : "border-hkc-line bg-hkc-surface"
            }`}
          >
            {highlighted && (
              <span className="absolute -top-2.5 left-4 inline-flex items-center gap-1 rounded-full border border-hkc-accent bg-hkc-bg px-2 py-0.5 font-mono text-[9.5px] tracking-widest text-hkc-accent uppercase">
                <Sparkles className="size-3" />
                {t.blRecommended}
              </span>
            )}
            <p className="text-[15px] text-hkc-ink">{p.name}</p>
            <p className="mt-1 font-mono text-lg text-hkc-accent">{p.priceLabel}</p>
            <p className="mt-2 flex-1 text-[12.5px] text-hkc-mut">{p.creditsLabel}</p>
            <Button
              onClick={() => onSelect(key)}
              variant={highlighted ? "default" : "outline"}
              className={
                highlighted
                  ? "mt-4 bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
                  : "mt-4 border-hkc-line bg-transparent text-hkc-ink hover:bg-hkc-hover"
              }
            >
              <CreditCard className="size-4" />
              {ctaLabel(key)}
            </Button>
          </div>
        );
      })}
    </div>
  );
}

export default function BillingPage() {
  const { creds } = useConsole();
  const isSelfHosted = creds.authMode === "key";
  const t = useAppT();
  const locale = useLocale();
  const dateLocale = locale === "fr" ? "fr-FR" : "en-US";
  const plans = cloudPlans(t);
  const { data, error, loading, reload } = useApi<BillingStatus>(
    isSelfHosted ? null : fetchBillingStatus,
    [isSelfHosted],
  );
  const [pendingPlan, setPendingPlan] = useState<PlanKey | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function subscribe() {
    if (!pendingPlan) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      // Sprint 17 (Dodo): no customer_phone — the hosted checkout collects
      // the payment method itself. The response's redirect_url IS the Dodo
      // hosted page; a same-page navigation is what the user expects.
      const response = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan: pendingPlan }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(body?.error?.message ?? `Erreur API (HTTP ${response.status})`);
      }
      if (body?.redirect_url) {
        window.location.href = body.redirect_url;
        return;
      }
      setPendingPlan(null);
      reload();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : t.ovUnknownError);
      setSubmitting(false);
    }
  }

  return (
    <div>
      <PageHeader title={t.navBilling} description={t.blPageDesc} />

      <div className="max-w-2xl space-y-5">
        {isSelfHosted && <SelfHostedNotice t={t} />}

        {!isSelfHosted && error && <ErrorState message={error} onRetry={reload} />}
        {!isSelfHosted && !error && loading && <Loading rows={2} />}

        {!isSelfHosted && !error && !loading && data && (
          <div className="space-y-4">
            {data.is_subscribed && (
              <div className="rounded-lg border border-hkc-line bg-hkc-surface px-5 py-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-mono text-[10px] tracking-widest text-hkc-faint uppercase">
                      {t.blCurrentPlan}
                    </p>
                    <p className="mt-0.5 text-lg text-hkc-ink">
                      {data.subscription_plan
                        ? (plans[data.subscription_plan as PlanKey]?.name ?? data.subscription_plan)
                        : t.blNone}
                    </p>
                  </div>
                  {data.subscription_status && <StatusBadge status={data.subscription_status} />}
                </div>
                <div className="mt-4 space-y-1 border-t border-hkc-line pt-4 text-[13px] text-hkc-mut">
                  <p>
                    {data.subscription_plan
                      ? plans[data.subscription_plan as PlanKey]?.creditsLabel
                      : null}
                  </p>
                  {data.current_period_end && (
                    <p>
                      {t.blNextRenewal}{" "}
                      <span className="text-hkc-ink">
                        {new Date(data.current_period_end).toLocaleDateString(dateLocale)}
                      </span>
                    </p>
                  )}
                </div>
              </div>
            )}

            {data.subscription_status === "pending" && (
              <div className="rounded-lg border border-hkc-line bg-hkc-surface px-5 py-5">
                <p className="text-[13px] text-hkc-mut">{t.blPendingNotice}</p>
                <Button
                  onClick={reload}
                  variant="outline"
                  className="mt-3 border-hkc-line bg-transparent text-hkc-ink hover:bg-hkc-hover"
                >
                  {t.blCheckStatus}
                </Button>
              </div>
            )}

            {!data.is_subscribed && data.subscription_status !== "pending" && (
              <div>
                <p className="mb-3 text-[13px] text-hkc-mut">
                  {data.subscription_status === "past_due" || data.subscription_status === "cancelled"
                    ? t.blResumeSubscription
                    : t.blChoosePlan}
                </p>
                <PlanCards
                  t={t}
                  onSelect={(key) => {
                    setSubmitError(null);
                    setPendingPlan(key);
                  }}
                  ctaLabel={() => t.blSubscribe}
                />
              </div>
            )}
          </div>
        )}

        {!isSelfHosted && <CreditsCard />}
      </div>

      <Dialog open={pendingPlan !== null} onOpenChange={(open) => !open && setPendingPlan(null)}>
        <DialogContent className="border-hkc-line bg-hkc-bg text-hkc-ink">
          <DialogHeader>
            <DialogTitle className="font-sans font-bold">{t.blConfirmSubscription}</DialogTitle>
            <DialogDescription className="text-hkc-mut">
              {pendingPlan && (
                <>
                  Plan {plans[pendingPlan].name} — {plans[pendingPlan].priceLabel}.{" "}
                  {t.blConfirmDialogMiddle}
                </>
              )}
            </DialogDescription>
          </DialogHeader>

          {/* Sprint 17 (Dodo): the hosted checkout collects the payment
              method itself — no phone number is collected or sent here. */}

          {submitError && (
            <p className="rounded-md border border-hkc-sup/40 bg-hkc-sup/10 px-3 py-2 text-[13px] text-hkc-sup">
              {submitError}
            </p>
          )}

          <DialogFooter>
            <Button
              onClick={subscribe}
              disabled={submitting}
              className="bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
            >
              <CreditCard className="size-4" />
              {submitting
                ? t.blRedirecting
                : pendingPlan
                  ? `${t.blSubscribe} — ${plans[pendingPlan].priceLabel}`
                  : t.blSubscribe}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
