"""Generates a high-resolution, perfectly routed transparent SVG and PNG architecture diagram for Certus.
Features larger, highly legible slide fonts and pure white text for the SHA-256 audit ledger.
"""
import os
import subprocess

svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1380 980" width="1380" height="980">
  <defs>
    <style>
      .node-text { font-family: 'Segoe UI', Inter, -apple-system, Roboto, sans-serif; font-size: 13.8px; fill: #1B2E18; font-weight: 500; text-anchor: middle; }
      .node-title { font-family: 'Segoe UI', Inter, -apple-system, Roboto, sans-serif; font-size: 15.5px; fill: #10210E; font-weight: 700; text-anchor: middle; }
      .edge-label { font-family: 'Segoe UI', Inter, -apple-system, Roboto, sans-serif; font-size: 13px; fill: #244220; font-weight: 700; text-anchor: middle; }
      .edge-path { fill: none; stroke: #4D6B45; stroke-width: 2.3; stroke-linecap: round; stroke-linejoin: round; }
      .edge-gold { fill: none; stroke: #B8771B; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#4D6B45" />
    </marker>
    <marker id="arrow-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#B8771B" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#C84832" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#2B6EA8" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#3D7D31" />
    </marker>
    <filter id="shadow" x="-4%" y="-4%" width="108%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="1" dy="2.2" stdDeviation="2.5" flood-color="#1B2E18" flood-opacity="0.10" />
    </filter>
  </defs>

  <!-- ================= SUBGRAPH CONTAINERS ================= -->
  <!-- Reader 1 Container -->
  <rect x="460" y="275" width="580" height="330" rx="16" ry="16" fill="#F4F8F1" stroke="#688C60" stroke-width="2.2" stroke-dasharray="7,5" filter="url(#shadow)" fill-opacity="0.95" />
  <rect x="485" y="260" width="330" height="30" rx="6" ry="6" fill="#587E50" />
  <text x="650" y="280" font-family="'Segoe UI', Inter, sans-serif" font-size="13.5px" font-weight="700" fill="#FFFFFF" text-anchor="middle">READER 1: CERTUS (EXPLAINS ITSELF)</text>

  <!-- Reader 2 Container -->
  <rect x="50" y="400" width="365" height="155" rx="14" ry="14" fill="#F4F8F1" stroke="#688C60" stroke-width="2" stroke-dasharray="7,5" filter="url(#shadow)" fill-opacity="0.95" />
  <rect x="75" y="386" width="295" height="28" rx="6" ry="6" fill="#587E50" />
  <text x="222" y="405" font-family="'Segoe UI', Inter, sans-serif" font-size="13px" font-weight="700" fill="#FFFFFF" text-anchor="middle">READER 2: WHOLE-IMAGE GRADER</text>

  <!-- ================= ROW 1: INPUT ================= -->
  <g filter="url(#shadow)">
    <rect x="510" y="16" width="400" height="70" rx="12" ry="12" fill="#EAF2E6" stroke="#4D6B45" stroke-width="2.3" />
    <text x="710" y="43" class="node-title" font-size="16">Fundus Photograph (Consent Recorded)</text>
    <text x="710" y="68" class="node-text" fill="#3D5936">FOV Normalised onto 1536 px Canvas</text>
  </g>

  <!-- ================= ROW 2: QUALITY GATE ================= -->
  <g filter="url(#shadow)">
    <polygon points="710,110 825,154 710,198 595,154" fill="#E2EDE0" stroke="#3E5E36" stroke-width="2.4" />
    <text x="710" y="149" class="node-title" font-size="15.5">Quality Gate</text>
    <text x="710" y="169" class="node-text" font-size="12.8">Adequacy Evaluator</text>
  </g>

  <!-- Retake Node (Left of Quality Gate) -->
  <g filter="url(#shadow)">
    <rect x="170" y="122" width="245" height="64" rx="10" ry="10" fill="#FDF3E7" stroke="#D97A26" stroke-width="2" />
    <text x="292" y="149" class="node-title" fill="#A04F08" font-size="15">Retake Image</text>
    <text x="292" y="171" class="node-text" fill="#874306" font-size="12.8">Technician Recapture Advice</text>
  </g>

  <!-- ================= ROW 3: READER 1 INTERNALS ================= -->
  <!-- Top: 9 Tiles + 1 Global View -->
  <g filter="url(#shadow)">
    <rect x="580" y="305" width="340" height="62" rx="10" ry="10" fill="#FFFFFF" stroke="#52734A" stroke-width="2" />
    <text x="750" y="331" class="node-title" font-size="15">9 Tiles (512 px) + 1 Global View</text>
    <text x="750" y="353" class="node-text" fill="#4C6E44">EfficientNet-B0 Encoder + U-Net Decoder</text>
  </g>

  <!-- Middle Left: Gated Attention -->
  <g filter="url(#shadow)">
    <rect x="490" y="398" width="230" height="68" rx="10" ry="10" fill="#FFFFFF" stroke="#52734A" stroke-width="2" />
    <text x="605" y="426" class="node-title" font-size="14.5">Gated Attention</text>
    <text x="605" y="449" class="node-text" font-size="12.8">Tile Pooling Weights</text>
  </g>

  <!-- Middle Right: Lesion & Vessel Masks + Evidence -->
  <g filter="url(#shadow)">
    <rect x="745" y="398" width="275" height="68" rx="10" ry="10" fill="#FFFFFF" stroke="#52734A" stroke-width="2" />
    <text x="882" y="424" class="node-title" font-size="14.5">Lesion, Disc &amp; Vessel Masks</text>
    <text x="882" y="448" class="node-text" font-size="12.5" fill="#B35212" font-weight="700">12-Number Evidence (Stop-Gradient)</text>
  </g>

  <!-- Bottom: Ordinal Grade Head -->
  <g filter="url(#shadow)">
    <rect x="545" y="505" width="410" height="66" rx="10" ry="10" fill="#EAF3E7" stroke="#3D5F35" stroke-width="2.2" />
    <text x="750" y="532" class="node-title" font-size="15.5">Ordinal Grade Head (ICDR 0–4)</text>
    <text x="750" y="555" class="node-text" font-size="13" fill="#3D5F35">Temperature Scaling (T=1.211) + Screening Band</text>
  </g>

  <!-- ================= ROW 3: READER 2 INTERNALS ================= -->
  <g filter="url(#shadow)">
    <rect x="75" y="435" width="315" height="66" rx="10" ry="10" fill="#FFFFFF" stroke="#52734A" stroke-width="2" />
    <text x="232" y="462" class="node-title" font-size="15">Whole Canvas EfficientNet-B0</text>
    <text x="232" y="485" class="node-text" font-size="12.8">512 px Baseline with Independent Band</text>
  </g>

  <!-- ================= ROW 4: DECISION & EXPLANATION ================= -->
  <!-- Camera Reliability Box -->
  <g filter="url(#shadow)">
    <rect x="65" y="640" width="235" height="66" rx="10" ry="10" fill="#F2F7F0" stroke="#587A50" stroke-width="1.8" />
    <text x="182" y="667" class="node-title" font-size="14.5">Camera Reliability</text>
    <text x="182" y="690" class="node-text" font-size="12.5">Dynamic Trust Table (Quality × Camera)</text>
  </g>

  <!-- Central Decision Diamond: Two-Reader Consensus -->
  <g filter="url(#shadow)">
    <polygon points="565,633 695,680 565,727 435,680" fill="#E2EDE0" stroke="#36582E" stroke-width="2.6" />
    <text x="565" y="674" class="node-title" font-size="15">Two-Reader Consensus</text>
    <text x="565" y="696" class="node-text" font-size="12.8" font-weight="700" fill="#244220">Both agree &amp; both sure?</text>
  </g>

  <!-- Explanation Module (Right side) -->
  <g filter="url(#shadow)">
    <rect x="910" y="622" width="425" height="92" rx="12" ry="12" fill="#FEF8EE" stroke="#C9882B" stroke-width="2.4" />
    <text x="1122" y="652" class="node-title" fill="#8C5708" font-size="15.5">Explainability &amp; Clinical Evidence</text>
    <text x="1122" y="676" class="node-text" fill="#693F02" font-size="12.8">Lesions (MA/HE/EX/SE), Disc/Fovea Geometry,</text>
    <text x="1122" y="697" class="node-text" fill="#693F02" font-size="12.8">Gated Tile Attention &amp; Exact Native Grad-CAM</text>
  </g>

  <!-- ================= ROW 5: OUTCOMES ================= -->
  <!-- 1. Auto-Refer -->
  <g filter="url(#shadow)">
    <rect x="325" y="795" width="190" height="62" rx="10" ry="10" fill="#FCECE8" stroke="#C84832" stroke-width="2.2" />
    <text x="420" y="823" class="node-title" fill="#A32410" font-size="15">Auto-Refer</text>
    <text x="420" y="844" class="node-text" fill="#801C0D" font-size="12.5">Specialist Alert Issued</text>
  </g>

  <!-- 2. Ophthalmologist Tele-Review -->
  <g filter="url(#shadow)">
    <rect x="550" y="783" width="320" height="78" rx="12" ry="12" fill="#EAF3FA" stroke="#2B6EA8" stroke-width="2.4" />
    <text x="710" y="812" class="node-title" fill="#154B78" font-size="15.5">Ophthalmologist Tele-Review</text>
    <text x="710" y="834" class="node-text" fill="#113F66" font-size="12.8">Triaged with Reason &amp; Evidence</text>
    <text x="710" y="852" class="node-text" fill="#0D3252" font-size="12" font-weight="700">Rapid Validation (&lt; 30 Seconds)</text>
  </g>

  <!-- 3. Auto-Clear -->
  <g filter="url(#shadow)">
    <rect x="910" y="795" width="230" height="62" rx="10" ry="10" fill="#E8F4E4" stroke="#3D7D31" stroke-width="2.2" />
    <text x="1025" y="823" class="node-title" fill="#205916" font-size="15">Auto-Clear (~80%)</text>
    <text x="1025" y="844" class="node-text" fill="#174510" font-size="12.5">Rescreen in 12 Months</text>
  </g>

  <!-- ================= ROW 6: AUDIT LEDGER (100% PURE WHITE TEXT) ================= -->
  <g filter="url(#shadow)">
    <rect x="410" y="895" width="560" height="52" rx="26" ry="26" fill="#1E331B" stroke="#122110" stroke-width="2.2" />
    <text x="690" y="928" font-family="'Segoe UI', Inter, -apple-system, sans-serif" font-size="15.5px" font-weight="700" fill="#FFFFFF" text-anchor="middle" letter-spacing="0.3px">
      SHA-256 Hash-Chained Audit Ledger (Tamper-Evident)
    </text>
  </g>

  <!-- ================= CONNECTING EDGES ================= -->

  <!-- Input to Quality Gate -->
  <path d="M 710 86 L 710 110" class="edge-path" marker-end="url(#arrow)" />

  <!-- Input to Reader 2 -->
  <path d="M 510 51 C 50 51 50 250 75 440" class="edge-path" marker-end="url(#arrow)" />

  <!-- Quality Gate -> Retake (straight left) -->
  <path d="M 595 154 L 415 154" class="edge-path" marker-end="url(#arrow)" />
  <rect x="470" y="141" width="65" height="22" rx="4" fill="#FFFFFF" stroke="#D97A26" stroke-width="1.2" />
  <text x="502" y="157" class="edge-label" fill="#A04F08">reject</text>

  <!-- Quality Gate -> Reader 1 Center: Usable (Adaptive CLAHE) -->
  <path d="M 710 198 L 710 305" class="edge-path" marker-end="url(#arrow)" />
  <rect x="610" y="213" width="200" height="24" rx="5" fill="#FFFFFF" stroke="#688C60" stroke-width="1.4" filter="url(#shadow)" />
  <text x="710" y="230" class="edge-label" fill="#244220">usable: adaptive CLAHE</text>

  <!-- Quality Gate -> Reader 1 Right: Good -->
  <path d="M 825 154 C 895 154 895 240 855 305" class="edge-path" marker-end="url(#arrow)" />
  <rect x="860" y="213" width="60" height="22" rx="4" fill="#FFFFFF" stroke="#688C60" stroke-width="1.2" filter="url(#shadow)" />
  <text x="890" y="229" class="edge-label" fill="#2A5021">good</text>

  <!-- Inside Reader 1: Tile to Gated Attention -->
  <path d="M 665 367 C 635 376 615 385 605 398" class="edge-path" marker-end="url(#arrow)" />

  <!-- Inside Reader 1: Tile to Lesion Masks -->
  <path d="M 835 367 C 865 376 875 385 882 398" class="edge-path" marker-end="url(#arrow)" />

  <!-- Inside Reader 1: Gated Attention to Grade Head -->
  <path d="M 615 466 C 635 487 665 497 685 505" class="edge-path" marker-end="url(#arrow)" />

  <!-- Inside Reader 1: Lesion Masks to Grade Head -->
  <path d="M 865 466 C 845 487 815 497 795 505" class="edge-path" marker-end="url(#arrow)" />

  <!-- Reader 1 Lesions to Explanation Module: Direct rightward route -->
  <path d="M 1020 432 C 1120 432 1120 540 1120 622" class="edge-gold" marker-end="url(#arrow-gold)" />

  <!-- Reliability to Consensus Diamond -->
  <path d="M 300 673 L 435 680" class="edge-path" marker-end="url(#arrow)" />

  <!-- Reader 1 Grade Head to Consensus Diamond -->
  <path d="M 720 571 C 700 600 650 630 610 650" class="edge-path" marker-end="url(#arrow)" />

  <!-- Reader 2 to Consensus Diamond -->
  <path d="M 245 501 C 260 560 380 630 455 660" class="edge-path" marker-end="url(#arrow)" />

  <!-- Decision Outcomes -->
  <!-- 1. Both Refer (Left) -->
  <path d="M 475 706 C 445 735 425 760 420 795" class="edge-path" stroke="#C84832" marker-end="url(#arrow-red)" />
  <rect x="395" y="740" width="82" height="22" rx="4" fill="#FFFFFF" stroke="#C84832" stroke-width="1.2" />
  <text x="436" y="756" class="edge-label" fill="#A32410">both refer</text>

  <!-- 2. Otherwise (Center) -->
  <path d="M 590 720 C 620 745 660 760 690 783" class="edge-path" stroke="#2B6EA8" marker-end="url(#arrow-blue)" />
  <rect x="635" y="741" width="82" height="22" rx="4" fill="#FFFFFF" stroke="#2B6EA8" stroke-width="1.2" />
  <text x="676" y="757" class="edge-label" fill="#154B78">otherwise</text>

  <!-- 3. Both Clear (Right) -->
  <path d="M 695 680 C 825 680 945 735 1015 795" class="edge-path" stroke="#3D7D31" marker-end="url(#arrow-green)" />
  <rect x="835" y="735" width="84" height="22" rx="4" fill="#FFFFFF" stroke="#3D7D31" stroke-width="1.2" />
  <text x="877" y="751" class="edge-label" fill="#205916">both clear</text>

  <!-- Explanation Module to Tele-Review -->
  <path d="M 1010 714 C 960 745 885 760 845 783" class="edge-gold" marker-end="url(#arrow-gold)" />

  <!-- Explanation to Auto-Refer (Dashed auxiliary) -->
  <path d="M 910 675 C 800 675 520 730 470 795" class="edge-gold" stroke-dasharray="5,4" marker-end="url(#arrow-gold)" />

  <!-- Outcomes into Audit Log -->
  <path d="M 420 857 C 420 885 520 895 560 895" class="edge-path" marker-end="url(#arrow)" />
  <path d="M 710 861 L 710 895" class="edge-path" marker-end="url(#arrow)" />
  <path d="M 1025 857 C 1025 885 850 895 810 895" class="edge-path" marker-end="url(#arrow)" />
</svg>
"""

svg_path = r"d:\Certus\docs\architecture_transparent.svg"
png_path = r"d:\Certus\docs\architecture_transparent.png"

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_content.strip())
print(f"Saved optimized SVG to {svg_path}")

# Render to transparent PNG via Edge
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--default-background-color=00000000",
    "--window-size=1380,980",
    f"--screenshot={png_path}",
    f"file:///{svg_path.replace(os.sep, '/')}"
]
subprocess.run(cmd, check=True)
print(f"Rendered clean transparent PNG to {png_path}")
