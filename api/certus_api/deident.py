"""Strip identifying metadata from an uploaded photograph before it is stored.

A fundus camera does not just write pixels. JPEGs routinely carry an EXIF block holding the
camera make, model and serial number, the exact capture timestamp, sometimes operator or
comment strings, and on anything with a radio, GPS coordinates. In a screening programme that
is patient data: a serial number plus a timestamp plus a location identifies who was
photographed where and when, without a name appearing anywhere.

Storage here is content-addressed and permanent -- blobs are written once and never rewritten --
so metadata that arrives is metadata that stays. Stripping has to happen on the way in.

This is lossless. JPEG is a sequence of marker segments and the identifying ones can be dropped
without touching the entropy-coded image data, so the pixels a clinician sees are bit-identical
to the pixels the camera produced. Re-encoding through an image library would also remove the
metadata, but it would recompress a medical image to do it, and a microaneurysm is five to
fifteen pixels across. PNG is handled the same way, by dropping text and EXIF chunks.

What this is not: DICOM de-identification. A DICOM file carries patient name, ID and birth date
in the header itself, and removing those correctly is a different job with its own standard
(PS3.15 Annex E). That is implemented in MATLAB, where dicomanon and the Medical Imaging
Toolbox do it properly -- see matlab/deidentifyDicom.m. DICOM uploads are detected here and
refused rather than silently stored with their headers intact.
"""
import struct

# APP1 is EXIF (and XMP), APP2 can hold ICC and MPF, APP13 is Photoshop/IPTC, COM is a free-text
# comment. APP0 is JFIF -- density and thumbnail information a decoder may want, no identity --
# so it stays. Anything unrecognised stays too: dropping segments we do not understand risks
# breaking the image, and the conservative failure here is to keep a byte, not lose a pixel.
JPEG_DROP = {0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
             0xEE, 0xEF, 0xFE}
PNG_DROP = {b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME"}


def is_dicom(data: bytes) -> bool:
    """DICOM Part 10: 128 preamble bytes then the magic 'DICM'."""
    return len(data) > 132 and data[128:132] == b"DICM"


def strip_metadata(data: bytes) -> tuple[bytes, dict]:
    """Return (clean_bytes, report). Unknown formats are passed through untouched and said so."""
    if data[:2] == b"\xff\xd8":
        return _strip_jpeg(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _strip_png(data)
    return data, {"format": "unknown", "stripped": [], "bytes_removed": 0}


def _strip_jpeg(data: bytes) -> tuple[bytes, dict]:
    out = bytearray(data[:2])
    dropped: list[str] = []
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            break                                    # not a marker boundary: stop editing, keep rest
        marker = data[i + 1]
        if marker == 0xDA:                           # start of scan; entropy data runs to the end
            out += data[i:]
            i = n
            break
        if 0xD0 <= marker <= 0xD9:                   # standalone markers carry no length field
            out += data[i:i + 2]
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        end = i + 2 + seg_len
        if seg_len < 2 or end > n:
            out += data[i:]                          # malformed: keep the remainder verbatim
            i = n
            break
        if marker in JPEG_DROP:
            dropped.append(_jpeg_name(marker, data[i + 4:min(end, i + 14)]))
        else:
            out += data[i:end]
        i = end
    else:
        out += data[i:]
    return bytes(out), {"format": "jpeg", "stripped": dropped,
                        "bytes_removed": len(data) - len(out)}


def _jpeg_name(marker: int, head: bytes) -> str:
    if marker == 0xFE:
        return "COM"
    tag = head.split(b"\x00")[0][:12].decode("ascii", "replace")
    return f"APP{marker - 0xE0}" + (f" ({tag})" if tag else "")


def _strip_png(data: bytes) -> tuple[bytes, dict]:
    out = bytearray(data[:8])
    dropped: list[str] = []
    i = 8
    n = len(data)
    while i + 8 <= n:
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        end = i + 12 + length                        # length + type + data + crc
        if end > n:
            out += data[i:]
            break
        if ctype in PNG_DROP:
            dropped.append(ctype.decode("ascii", "replace"))
        else:
            out += data[i:end]
        i = end
        if ctype == b"IEND":
            break
    return bytes(out), {"format": "png", "stripped": dropped,
                        "bytes_removed": len(data) - len(out)}
