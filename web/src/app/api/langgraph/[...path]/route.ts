const DEFAULT_AGENT_SERVER_URL = "http://127.0.0.1:2024";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxy(request: Request, context: RouteContext) {
  const { path } = await context.params;
  const baseUrl = (
    process.env.LANGGRAPH_API_URL ?? DEFAULT_AGENT_SERVER_URL
  ).replace(/\/$/, "");
  const upstream = new URL(
    `${baseUrl}/${path.map(encodeURIComponent).join("/")}`,
  );
  upstream.search = new URL(request.url).search;

  const headers = new Headers(request.headers);
  for (const name of [
    "host",
    "connection",
    "content-length",
    "cookie",
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
  ]) {
    headers.delete(name);
  }
  const apiKey = process.env.LANGGRAPH_API_KEY?.trim();
  if (apiKey) headers.set("x-api-key", apiKey);

  const hasBody = !["GET", "HEAD"].includes(request.method);
  const response = await fetch(upstream, {
    method: request.method,
    headers,
    body: hasBody ? await request.arrayBuffer() : undefined,
    cache: "no-store",
    redirect: "manual",
  });

  const responseHeaders = new Headers(response.headers);
  for (const name of [
    "connection",
    "content-encoding",
    "content-length",
    "set-cookie",
    "transfer-encoding",
  ]) {
    responseHeaders.delete(name);
  }
  responseHeaders.set("x-accel-buffering", "no");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
export const HEAD = proxy;
