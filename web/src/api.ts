/* The one place the frontend talks to the backend.

   Every call carries the API key as X-API-Key, the header `security.py` expects. A failure to reach
   the server and a rejection by the server are different problems with different fixes, so they come
   back as different statuses rather than one generic "something went wrong". */
import { getSession } from "./auth";

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  /** HTTP status, or 0 when the request never reached the server. */
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }

  get unreachable() {
    return this.status === 0;
  }
}

type Options = Omit<RequestInit, "body"> & { body?: unknown; key?: string };

export async function api<T>(path: string, opts: Options = {}): Promise<T> {
  const { body, key, headers, ...rest } = opts;
  const apiKey = key ?? getSession()?.key ?? "";

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...rest,
      headers: {
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
        ...headers,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}`);
  }

  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; fall back to the status line if it is not JSON.
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (typeof j?.detail === "string") detail = j.detail;
    } catch { /* not JSON, keep the status line */ }
    throw new ApiError(res.status, detail);
  }

  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

/** Liveness only. Deliberately not /readyz: that one loads the checkpoint onto the GPU, which is
    not something a sign-in screen should trigger while a training run is using the card. */
export function ping() {
  return api<{ status: string }>("/healthz");
}

/* Images and lesion masks are behind the same API key as everything else, and an <img src> cannot
   send a header. So fetch the bytes with the key and hand the element an object URL instead. */
export async function authedBlob(path: string): Promise<string> {
  const key = getSession()?.key ?? "";
  const res = await fetch(`${API_BASE}${path}`, { headers: key ? { "X-API-Key": key } : {} });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return URL.createObjectURL(await res.blob());
}

/** Multipart upload. FormData sets its own Content-Type boundary, so it cannot go through `api`. */
export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const key = getSession()?.key ?? "";
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: key ? { "X-API-Key": key } : {},
      body: form,
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (typeof j?.detail === "string") detail = j.detail;
    } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}
