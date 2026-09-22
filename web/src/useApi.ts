/* One GET, its error, and a way to ask again. Most console panels are exactly this, and writing the
   same six lines of state in every page is how the error handling drifts apart between them. */
import { useCallback, useEffect, useState } from "react";

import { ApiError, api } from "./api";

export function useApi<T>(path: string | null, auto = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    try {
      setData(await api<T>(path));
      setError(null);
    } catch (e) {
      const err = e as ApiError;
      setError(err.unreachable ? err.message : `${err.status}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    if (auto) reload();
  }, [reload, auto]);

  return { data, error, loading, reload };
}
