"use client";

import { Check, Copy, KeyRound, Plus } from "lucide-react";
import { useState } from "react";
import { useConsole } from "@/components/console/provider";
import {
  EmptyState,
  ErrorState,
  Loading,
  PageHeader,
} from "@/components/console/states";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { haki, HakiApiError, isUnauthorized } from "@/lib/api";
import { useAppT } from "@/lib/app/translations";
import { useLocale } from "@/lib/i18n/use-locale";
import { useApi } from "@/lib/use-api";
import { ApiKeyOut, KeyCreated } from "@/lib/types";

export default function KeysPage() {
  const { creds, logout } = useConsole();
  const { projectId, orgId } = creds;
  const t = useAppT();
  const locale = useLocale();
  const dateLocale = locale === "fr" ? "fr-FR" : "en-US";
  const [createOpen, setCreateOpen] = useState(false);
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [created, setCreated] = useState<KeyCreated | null>(null);
  const [copied, setCopied] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState<ApiKeyOut | null>(null);
  const [revoking, setRevoking] = useState(false);
  const [revokeError, setRevokeError] = useState<string | null>(null);

  const { data, error, loading, reload } = useApi<{ keys: ApiKeyOut[] }>(
    () => haki<{ keys: ApiKeyOut[] }>("v1/keys"),
    [projectId],
  );

  async function createKey() {
    setCreating(true);
    setCreateError(null);
    try {
      const result = await haki<KeyCreated>("v1/keys", {
        method: "POST",
        body: {
          org_id: orgId,
          project_id: projectId,
          label: label.trim() || null,
        },
      });
      setCreated(result);
      setCreateOpen(false);
      setLabel("");
      reload();
    } catch (err) {
      if (isUnauthorized(err)) {
        setCreateError(t.kyCreateDenied);
      } else if (err instanceof HakiApiError) {
        setCreateError(err.message);
      } else {
        setCreateError(t.ovUnknownError);
      }
    } finally {
      setCreating(false);
    }
  }

  async function confirmRevoke() {
    if (!revokeTarget) return;
    setRevoking(true);
    setRevokeError(null);
    try {
      await haki(`v1/keys/${revokeTarget.id}`, { method: "DELETE" });
      setRevokeTarget(null);
      reload();
    } catch (err) {
      if (isUnauthorized(err)) {
        logout();
        return;
      }
      // Keep the dialog open with an explicit error: silently closing it
      // made a failed revocation look successful — the key would stay
      // active while the user believes it is dead.
      setRevokeError(
        err instanceof HakiApiError ? err.message : t.ovUnknownError,
      );
    } finally {
      setRevoking(false);
    }
  }

  async function copyKey() {
    if (!created) return;
    await navigator.clipboard.writeText(created.key);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div>
      <PageHeader title={t.navKeys} description={t.kyDesc} />

      {created && (
        <div className="mb-5 rounded-lg border border-hkc-accent/40 bg-hkc-chip px-4 py-4">
          <p className="flex items-center gap-2 text-sm text-hkc-accent">
            <KeyRound className="size-4" /> {t.kyCreatedNotice}
          </p>
          <div className="mt-2.5 flex items-center gap-2">
            <code className="flex-1 overflow-x-auto rounded-md border border-hkc-line bg-hkc-bg px-3 py-2 font-mono text-[12px] text-hkc-ink">
              {created.key}
            </code>
            <Button
              onClick={copyKey}
              className="bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
            >
              {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
              {copied ? t.kyCopied : t.kyCopy}
            </Button>
          </div>
        </div>
      )}

      <div className="mb-4 flex justify-end">
        <Button
          onClick={() => setCreateOpen(true)}
          disabled={!orgId}
          className="bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
        >
          <Plus className="size-4" /> {t.kyNewKeyBtn}
        </Button>
      </div>
      {!orgId && (
        <p className="mb-4 text-right font-mono text-[11px] text-hkc-faint">
          {t.kyNeedOrgHint}
        </p>
      )}

      {error && <ErrorState message={error} onRetry={reload} />}
      {!error && loading && <Loading rows={3} />}
      {!error && !loading && data && data.keys.length === 0 && (
        <EmptyState title={t.kyNoKeys} />
      )}
      {!error && !loading && data && data.keys.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-hkc-line">
          <table className="w-full min-w-[680px] text-left">
            <thead>
              <tr className="border-b border-hkc-line bg-hkc-surface font-mono text-[10.5px] tracking-widest text-hkc-faint uppercase">
                <th className="px-3 py-2.5 font-medium">{t.kyColPrefix}</th>
                <th className="px-3 py-2.5 font-medium">Label</th>
                <th className="px-3 py-2.5 font-medium">{t.kyColProject}</th>
                <th className="px-3 py-2.5 font-medium">{t.kyColCreatedOn}</th>
                <th className="px-3 py-2.5 font-medium">{t.mmStatusLabel}</th>
                <th className="px-3 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {data.keys.map((key) => (
                <tr key={key.id} className="border-b border-hkc-line last:border-0">
                  <td className="px-3 py-3 font-mono text-[12.5px] text-hkc-ink">
                    {key.prefix}…
                  </td>
                  <td className="px-3 py-3 text-[13px] text-hkc-mut">
                    {key.label ?? "—"}
                  </td>
                  <td className="px-3 py-3 font-mono text-[12px] text-hkc-mut">
                    {key.project_id}
                  </td>
                  <td className="px-3 py-3 font-mono text-[11.5px] text-hkc-mut">
                    {new Date(key.created_at).toLocaleDateString(dateLocale)}
                  </td>
                  <td className="px-3 py-3">
                    {key.revoked_at ? (
                      <span className="rounded-sm border border-hkc-line bg-hkc-hover px-1.5 py-0.5 font-mono text-[10.5px] text-hkc-faint uppercase">
                        {t.kyRevoked}
                      </span>
                    ) : (
                      <span className="rounded-sm border border-hkc-ok/30 bg-hkc-ok/10 px-1.5 py-0.5 font-mono text-[10.5px] text-hkc-ok uppercase">
                        {t.kyActive}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-3 text-right">
                    {!key.revoked_at && (
                      <button
                        type="button"
                        onClick={() => setRevokeTarget(key)}
                        className="rounded-sm border border-hkc-line px-2 py-1 font-mono text-[11px] text-hkc-mut transition-colors hover:border-hkc-sup/40 hover:text-hkc-sup"
                      >
                        {t.kyRevokeBtn}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-hkc-line bg-hkc-surface text-hkc-ink">
          <DialogHeader>
            <DialogTitle className="font-sans font-bold">{t.kyCreateDialogTitle}</DialogTitle>
            <DialogDescription className="text-hkc-mut">
              {t.kyLinkedToProject}{" "}
              <code className="font-mono text-[12px]">{projectId}</code>
              {orgId && (
                <>
                  {" "}
                  · {t.kyOrgSuffix}{" "}
                  <code className="font-mono text-[12px]">{orgId}</code>
                </>
              )}
              .
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="label" className="text-hkc-mut">
              {t.kyLabelOptional}
            </Label>
            <Input
              id="label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="console, ci, cursor…"
              className="border-hkc-line bg-hkc-bg text-hkc-ink placeholder:text-hkc-faint"
            />
          </div>
          {createError && (
            <p className="rounded-md border border-hkc-sup/40 bg-hkc-sup/10 px-3 py-2 text-[13px] text-hkc-sup">
              {createError}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateOpen(false)}
              className="border-hkc-line bg-transparent text-hkc-ink hover:bg-hkc-hover"
            >
              {t.mmCancel}
            </Button>
            <Button
              onClick={createKey}
              disabled={creating}
              className="bg-hkc-accent text-[#171208] hover:bg-hkc-accent"
            >
              {creating ? t.kyCreating : t.kyCreate}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={revokeTarget !== null}
        onOpenChange={(open) => !open && setRevokeTarget(null)}
      >
        <DialogContent className="border-hkc-line bg-hkc-surface text-hkc-ink">
          <DialogHeader>
            <DialogTitle className="font-sans font-bold">{t.kyRevokeDialogTitle}</DialogTitle>
            <DialogDescription className="text-hkc-mut">
              <code className="font-mono text-[12px]">
                {revokeTarget?.prefix}…
              </code>{" "}
              {t.kyRevokeDialogBody}
            </DialogDescription>
          </DialogHeader>
          {revokeError && (
            <p className="text-sm text-red-500" role="alert">
              {revokeError}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRevokeTarget(null)}
              className="border-hkc-line bg-transparent text-hkc-ink hover:bg-hkc-hover"
            >
              {t.mmCancel}
            </Button>
            <Button
              onClick={confirmRevoke}
              disabled={revoking}
              className="bg-hkc-sup/90 text-white hover:bg-hkc-sup"
            >
              {revoking ? t.kyRevoking : t.kyRevokeBtn}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
