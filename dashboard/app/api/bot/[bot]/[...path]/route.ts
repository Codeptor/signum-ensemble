import { NextRequest, NextResponse } from "next/server";

// Run in Mumbai (closest to Bangalore VPS servers)
export const runtime = "nodejs";
export const preferredRegion = "bom1";

const BOT_URLS: Record<string, string> = {
  "bot-a": process.env.BOT_A_URL!,
  "bot-b": process.env.BOT_B_URL!,
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bot: string; path: string[] }> }
) {
  const { bot, path } = await params;
  const baseUrl = BOT_URLS[bot];

  if (!baseUrl) {
    return NextResponse.json(
      { error: `Unknown bot: ${bot}` },
      { status: 400 }
    );
  }

  const endpoint = `/${path.join("/")}`;
  const searchParams = request.nextUrl.searchParams.toString();
  const url = `${baseUrl}${endpoint}${searchParams ? `?${searchParams}` : ""}`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);

    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });

    clearTimeout(timeout);

    if (!response.ok) {
      return NextResponse.json(
        { error: `Bot returned ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to reach bot";
    return NextResponse.json(
      { error: message, bot, endpoint },
      { status: 502 }
    );
  }
}
