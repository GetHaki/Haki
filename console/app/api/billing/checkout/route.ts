import { NextRequest, NextResponse } from "next/server";
import { auth, currentUser } from "@clerk/nextjs/server";

// Server-only: creates the subscription checkout for the signed-in user's
// organization. Same trust model as app/api/provision — the browser never
// sees HAKI_CONSOLE_SERVICE_KEY, and the backend endpoint
// (POST /v1/billing/checkout) only accepts that secret, never a customer
// hk_ key. This is a WRITE call with a REAL financial effect against the
// billing provider's live API: it only runs when a signed-in human
// explicitly submits the billing form (app/app/billing/page.tsx), never
// automatically.
//
// Sprint 17 (Dodo): the hosted checkout collects the payment method (and
// phone, if the method needs one) itself — customer_phone is OPTIONAL and
// merely forwarded when the client sends it (the legacy GeniusPay path
// still requires it server-side).
const API_URL = (
  process.env.NEXT_PUBLIC_HAKI_API_URL ?? "http://localhost:8100"
).replace(/\/$/, "");

interface CheckoutBody {
  customer_phone?: string;
  plan?: "starter" | "growth" | "scale";
}

export async function POST(request: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json(
      { error: { type: "unauthorized", message: "not signed in" } },
      { status: 401 },
    );
  }

  const serviceKey = process.env.HAKI_CONSOLE_SERVICE_KEY;
  if (!serviceKey) {
    return NextResponse.json(
      {
        error: {
          type: "billing_disabled",
          message: "HAKI_CONSOLE_SERVICE_KEY is not configured on the console backend",
        },
      },
      { status: 503 },
    );
  }

  let body: CheckoutBody;
  try {
    body = (await request.json()) as CheckoutBody;
  } catch {
    return NextResponse.json(
      { error: { type: "invalid_payload", message: "malformed JSON body" } },
      { status: 422 },
    );
  }

  const clerkUser = await currentUser();
  const name = clerkUser?.fullName || undefined;
  const email = clerkUser?.primaryEmailAddress?.emailAddress || undefined;

  let upstream: Response;
  try {
    upstream = await fetch(`${API_URL}/v1/billing/checkout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${serviceKey}`,
      },
      body: JSON.stringify({
        owner_ref: userId,
        // Optional since Dodo: the hosted checkout collects the payment
        // method itself. Forwarded only when present (the GeniusPay path
        // still validates it server-side).
        ...(body.customer_phone ? { customer_phone: body.customer_phone } : {}),
        customer_name: name,
        customer_email: email,
        ...(body.plan ? { plan: body.plan } : {}),
      }),
    });
  } catch {
    return NextResponse.json(
      {
        error: {
          type: "billing_unreachable",
          message: "L'API Haki est injoignable. Vérifiez sa configuration.",
        },
      },
      { status: 502 },
    );
  }

  const responseBody = await upstream.text();
  return new NextResponse(responseBody, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
