"""First-run catalogue: the camps and cameras a demo needs to exist before anything can be captured.

Startup only ran create_all, which makes empty tables. Sites and devices were therefore whatever
someone had POSTed to /v1/sites at some point on that machine, and the database is not in version
control -- so a fresh clone came up with an empty Site dropdown and no way to open a visit. That
is not a state the app can recover from through the UI a technician sees.

Seeding runs only when the catalogue is empty, so it never overwrites real sites, and deleting the
database is a safe way to start over.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Device, Site

# One calibrated camera and one that is not, on purpose. The trust layer discounts an eye
# photographed on a camera the calibration has never seen, and a demo catalogue where every camera
# is identical can never show that happening. Topcon_TRC_NW6 is a domain the trust layer actually
# has a fit for (it is one of the Messidor-2 cameras); remidio_fop is not, so the handheld reads
# UNVERIFIED and its eyes are trusted less -- which is the real situation for a handheld in a
# camp, not a contrivance.
CAMPS = [
    # domain_key must match a key in trust.json's per_domain table (Engine._load_calibration ->
    # _domain_table). APTOS_mixed_India is fitted there, so this camera reads VERIFIED (1.0 factor).
    # Topcon_TRC_NW6 only appears in external_messidor2 evaluation, not per_domain, so it was
    # silently unverified and the intended discount contrast was invisible.
    {"name": "Sitapur PHC", "district": "Sitapur", "state": "Uttar Pradesh", "kind": "phc",
     "camera": {"make": "Topcon", "model": "TRC-NW6", "serial": "TP-NW6-4471",
                "domain_key": "APTOS_mixed_India"}},
    # remidio_fop has no per_domain fit → reads UNVERIFIED (0.80 camera factor). Keeping this
    # asymmetry is the whole point of the two-camera seed: the trust discount is visible on screen.
    {"name": "Barabanki camp", "district": "Barabanki", "state": "Uttar Pradesh", "kind": "camp",
     "camera": {"make": "Remidio", "model": "FOP NM-10", "serial": "RM-10-5108",
                "domain_key": "remidio_fop"}},
]


def seed_if_empty(db: Session) -> int:
    """Create the demo catalogue when there are no sites. Returns how many sites were added."""
    if db.execute(select(Site.id).limit(1)).first():
        return 0
    for camp in CAMPS:
        cam = camp["camera"]
        site = Site(**{k: v for k, v in camp.items() if k != "camera"})
        db.add(site)
        db.flush()                       # need site.id before the camera can point at it
        db.add(Device(site_id=site.id, **cam))
    db.commit()
    return len(CAMPS)
