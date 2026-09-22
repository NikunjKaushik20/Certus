/* An <img> whose bytes need an API key. Revokes the object URL when the path changes or the
   component goes away, so a long review session does not leak every mask it has looked at.

   `version` forces a refetch when the path has not changed but the bytes behind it have. An image
   requested before grading returns 409 (not preprocessed yet); without this, that failure stuck to
   the URL and the panel stayed empty even after the visit was graded. */
import { useEffect, useState } from "react";

import { authedBlob } from "./api";

export function useAuthedImage(path: string | null, version: unknown = 0) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!path) { setUrl(null); return; }
    let dead = false;
    let made: string | null = null;
    authedBlob(path)
      .then((u) => {
        if (dead) { URL.revokeObjectURL(u); return; }
        made = u;
        setUrl(u);
      })
      .catch(() => !dead && setUrl(null));
    return () => {
      dead = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [path, version]);

  return url;
}
