/* The landing page. One fixed 3D stage, twelve sections of copy scrolling over it.

   Every number on this page is either measured in our own evaluation or cited to a source. Where a
   figure is an assumption it says so. That discipline is the product's whole argument, so the
   marketing page had better hold to it too. */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { Scene } from "../scene/Scene";
import { depthLabel, onProgress, startScroll } from "../scroll";
import "../index.css";

/* Measured on held-out data. Single place to update after a training run. */
/* Measured, not claimed. Every figure here is read from the shipped checkpoint
   (runs_torch/train_20260912_211356/best.pt, EMA weights at step 16,760) and the trust layer
   calibrate.py fitted beside it. val is the split the checkpoint was selected on, so it is the
   optimistic one and is labelled as such; test and Messidor-2 are the honest numbers, and
   Messidor-2 was never touched in training, selection or calibration.
   Update these in one pass whenever a new checkpoint is calibrated -- never by hand. */
const M = {
  model: "v0.4 · EfficientNet-B0 multi-task",
  trainEyes: "26,809",
  val: { n: "3,049", auc: "0.983", sens: "98.3%" },
  test: { n: "4,511", auc: "0.952", sens: "91.5%" },
  ext: { n: "1,740", auc: "0.952", sens: "90.1%", ece: "0.093", eceRaw: "0.083" },
  /* after temperature scaling (T = 1.211) and the screening band, from trust.json */
  band: {
    test: { ece: "0.079", eceRaw: "0.090", sysSens: "94.4%", abstain: "11.7%", missed: "118" },
    ext: { sysSens: "99.6%", abstain: "26.4%", missed: "2" },
  },
  /* the deployed rule: Certus and the whole-image grader must agree (trust.json -> second_reader) */
  two: {
    test: { sysSens: "97.6%", sysSpec: "95.2%", toHuman: "20.1%", missed: "50" },
    ext: { sysSens: "100%", sysSpec: "89.1%", toHuman: "45.9%" },
    extRefit: { patients: "100", sysSens: "97.4%", sysSpec: "96.5%", toHuman: "26.9%" },
  },
  eceByCamera: { aptos: "0.032", ddr: "0.094", idrid: "0.051" },
  quality: { acc: "91.8%", auc: "0.993" },
  dice: { ma: "0.43", he: "0.60", ex: "0.60", se: "0.51" },
  thr: { ma: "0.406", he: "0.648", ex: "0.828", se: "0.121" },
};

/* Ground truth for the retina in the scene: IDRiD image 79, expert-annotated. */
const SCENE_EYE = {
  ma: 76, he: 28, ex: 302,
  quadrants: [
    { q: "Superior temporal", ma: 18, he: 8, ex: 50 },
    { q: "Superior nasal", ma: 14, he: 5, ex: 51 },
    { q: "Inferior temporal", ma: 17, he: 8, ex: 73 },
    { q: "Inferior nasal", ma: 27, he: 7, ex: 128 },
  ],
};

function Chrome() {
  const [p, setP] = useState(0);
  useEffect(() => onProgress(setP), []);
  // Inside the eye the stage is almost black, so the fixed chrome has to invert or it disappears.
  // One class toggle, not a per-frame style write.
  useEffect(() => {
    document.body.classList.toggle("inside", p > 0.265 && p < 0.9);
  }, [p]);
  // the class is on <body>, which outlives this route -- leaving the page mid-scroll would otherwise
  // strand the whole console in the dark-interior palette
  useEffect(() => () => document.body.classList.remove("inside"), []);
  const [where, depth] = depthLabel(p);
  return (
    <>
      <div className="topbar">
        <div className="wordmark">Cert<span>us</span></div>
        <Link className="signin-link" to="/signin">Sign in</Link>
      </div>
      <div className="depth">
        {where}
        <b>{depth}</b>
      </div>
      <div className="progress">
        <i style={{ transform: `scaleX(${p})` }} />
      </div>
    </>
  );
}

export default function Landing() {
  useEffect(() => startScroll(), []);

  return (
    <>
      <Scene />
      <Chrome />

      <main className="scroll">
        {/* 1 ------------------------------------------------------------- */}
        <section>
          <div className="hero">
            <h1 className="hero-title">Certus</h1>
            <div className="hero-sub">Screening that shows its work.</div>
            <p className="lede dim">
              Caught early, diabetic retinopathy is treatable. Caught late, it blinds. Certus grades a
              retinal photograph in seconds, shows the lesions behind the grade, and hands the case to
              a person when it isn't sure.
            </p>
            <div className="scroll-cue">Scroll to look closer ↓</div>
          </div>
        </section>

        {/* 2 ------------------------------------------------------------- */}
        <section>
          <div className="card">
            <div className="eyebrow">The problem</div>
            <h2>Sight is lost between the screening and the clinic.</h2>
            <p>
              India's largest population-based study of people with diabetes found retinopathy in one
              in eight of them. One in twenty-five had already reached the sight-threatening stage.
              None of them would have felt it coming. The disease is silent until it is advanced,
              which is the whole reason screening exists.
            </p>
            <div className="grid">
              <div><dt>Have retinopathy</dt><dd className="sienna">12.5%<small>of people with diabetes</small></dd></div>
              <div><dt>Sight-threatening</dt><dd className="sienna">4.0%<small>vision-threatening DR</small></dd></div>
              <div><dt>Never reached care</dt><dd className="sienna">~90%<small>of those found with STDR</small></dd></div>
            </div>
            <p>
              The third number is the one to sit with. Finding the disease was never the bottleneck.
              Closing the loop is. A programme that reports back days later, to someone who has
              already travelled home, mostly produces paperwork.
            </p>
            <div className="src">
              SMART India study, <i>Lancet Global Health</i> 2022; follow-up treatment-seeking,
              <i> PLOS One</i> 2022.
            </div>
          </div>
        </section>

        {/* 3 ------------------------------------------------------------- */}
        <section className="right">
          <div className="card">
            <div className="eyebrow">Before anything else</div>
            <h2>A model that grades an unusable photograph is worse than no model.</h2>
            <p>
              In the field, a lot of fundus photographs cannot be graded by anyone. Too dark, out of
              focus, or blocked by a cataract or a pupil that would not dilate. How often that happens
              depends mostly on who is holding the camera.
            </p>
            <table className="data">
              <thead>
                <tr><th>Setting</th><th style={{ textAlign: "right" }}>Ungradable</th></tr>
              </thead>
              <tbody>
                <tr><td>Population screening, no quality handling</td><td className="n">22.5%</td></tr>
                <tr><td>Clinic deployment, Thailand</td><td className="n">21%</td></tr>
                <tr className="hi"><td>Handheld camera, trained operator</td><td className="n">~4%</td></tr>
              </tbody>
            </table>
            <p>
              So Certus judges the photograph before it grades anything, and asks for a retake while
              the patient is still in the chair. Only gradable images reach the grader.
            </p>
            <div className="grid">
              <div><dt>Gate accuracy</dt><dd>{M.quality.acc}<small>three classes</small></dd></div>
              <div><dt>Spotting unusable</dt><dd>{M.quality.auc}<small>AUC</small></dd></div>
            </div>
            <div className="src">SMART India 2022; Beede et al., CHI 2020; Remidio primary-care study.</div>
          </div>
        </section>

        {/* 4 ------------------------------------------------------------- */}
        <section>
          <div className="card narrow">
            <div className="eyebrow">Through the pupil</div>
            <h2>What you are looking at is real.</h2>
            <p>
              The retina ahead is not a rendering. It is a photograph from IDRiD, the Indian Diabetic
              Retinopathy Image Dataset. The lesions about to appear on it were outlined by
              ophthalmologists, pixel by pixel.
            </p>
            <div className="note">
              <b>Why the working scale matters.</b> A microaneurysm is five to fifteen pixels across
              on the sensor. Shrink the whole retina to the 512&nbsp;px a network usually expects and
              it is about one pixel: gone. Certus normalises the field to a 1536&nbsp;px canvas and
              reads it as nine 512&nbsp;px tiles plus one whole-eye view, where that same lesion is
              still three to seven pixels across.
            </div>
          </div>
        </section>

        {/* 5 ------------------------------------------------------------- */}
        <section className="right">
          <div className="card">
            <div className="eyebrow">Evidence</div>
            <h2>Every lesion it counts, it can point to.</h2>
            <p>
              These are the expert annotations for the eye in view. Certus produces the same
              quantities for a photograph it has never seen, which is what lets a reviewer check it
              instead of trusting it.
            </p>
            <div className="grid">
              <div><dt>Microaneurysms</dt><dd className="sienna">{SCENE_EYE.ma}</dd></div>
              <div><dt>Haemorrhages</dt><dd className="sienna">{SCENE_EYE.he}</dd></div>
              <div><dt>Hard exudates</dt><dd className="teal">{SCENE_EYE.ex}</dd></div>
            </div>
            <table className="data">
              <thead>
                <tr><th>Quadrant</th><th style={{ textAlign: "right" }}>MA</th><th style={{ textAlign: "right" }}>HE</th><th style={{ textAlign: "right" }}>EX</th></tr>
              </thead>
              <tbody>
                {SCENE_EYE.quadrants.map((q) => (
                  <tr key={q.q}><td>{q.q}</td><td className="n">{q.ma}</td><td className="n">{q.he}</td><td className="n">{q.ex}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="small">
              The quadrant split is not decoration. The 4-2-1 rule that defines severe
              non-proliferative disease is counted per quadrant, and which side a lesion falls on is
              worked out from where the optic disc actually sits, not assumed.
            </p>
            <div className="legend">
              <span><i style={{ background: "#a94e28" }} />Microaneurysms</span>
              <span><i style={{ background: "#7c3418" }} />Haemorrhages</span>
              <span><i style={{ background: "#b8925a" }} />Hard exudates</span>
            </div>
          </div>
        </section>

        {/* 6 ------------------------------------------------------------- */}
        <section>
          <div className="card">
            <div className="eyebrow">Not a heatmap</div>
            <h2>The grade is built from the findings, not explained after the fact.</h2>
            <p>
              Most explainable systems grade first and draw the heatmap afterwards. A heatmap shows
              where the network looked. It does not show what the network concluded. Certus feeds
              twelve measured quantities from the lesion maps into the grading head: how much of each
              lesion type is there, how strongly, and where it sits across the quadrants.
            </p>
            <p>
              So the explanation is an input to the decision rather than a picture drawn after it.
              Change the evidence and the grade changes with it. A clinician can test that.
            </p>
            <hr className="rule" />
            <h3>Thresholds are data, not code</h3>
            <p className="small">
              One cut-off cannot serve every lesion. Microaneurysms are faint and tiny; exudates are
              bright and large. Each threshold is fitted on held-out images and stored alongside the
              model, so recalibrating later never silently rewrites a report someone has already acted
              on.
            </p>
            <div className="grid">
              <div><dt>MA</dt><dd>{M.thr.ma}<small>dice {M.dice.ma}</small></dd></div>
              <div><dt>HE</dt><dd>{M.thr.he}<small>dice {M.dice.he}</small></dd></div>
              <div><dt>EX</dt><dd>{M.thr.ex}<small>dice {M.dice.ex}</small></dd></div>
              <div><dt>SE</dt><dd>{M.thr.se}<small>dice {M.dice.se}</small></dd></div>
            </div>
            <p className="small">
              Dice is measured on the test set with the thresholds fitted on validation. That is the
              honest transfer, not the flattering one. It is also brutal on small lesions by
              construction: shift a ten-pixel microaneurysm by two pixels and the score halves, even
              though you found it.
            </p>
          </div>
        </section>

        {/* 7 ------------------------------------------------------------- */}
        <section className="right">
          <div className="card">
            <div className="eyebrow">The grade</div>
            <h2>Five steps, one line that matters.</h2>
            <p>
              Certus grades on the international scale, 0 to 4. The question that changes what happens
              to the patient is narrower than that. Is this eye referable, grade 2 or worse? The model
              is built around that boundary instead of treating five grades as five unrelated labels.
            </p>
            <table className="data">
              <thead><tr><th>Grade</th><th>Meaning</th><th>Action</th></tr></thead>
              <tbody>
                <tr><td className="n">0</td><td>No retinopathy</td><td>Rescreen in a year</td></tr>
                <tr><td className="n">1</td><td>Mild</td><td>Rescreen</td></tr>
                <tr className="hi"><td className="n">2</td><td>Moderate</td><td>Refer</td></tr>
                <tr className="hi"><td className="n">3</td><td>Severe</td><td>Refer, sooner</td></tr>
                <tr className="hi"><td className="n">4</td><td>Proliferative</td><td>Urgent</td></tr>
              </tbody>
            </table>
            <div className="grid">
              <div><dt>Validation · {M.val.n}</dt><dd>{M.val.auc}<small>{M.val.sens} sensitivity</small></dd></div>
              <div><dt>Test · {M.test.n}</dt><dd>{M.test.auc}<small>{M.test.sens} sensitivity</small></dd></div>
              <div><dt>Unseen cameras · {M.ext.n}</dt><dd className="teal">{M.ext.auc}<small>{M.ext.sens} sensitivity</small></dd></div>
            </div>
            <p className="small">
              Sensitivity is quoted at 85% specificity. All three splits are here on purpose.
              Validation is the easy one, and it also picks the checkpoint, so it flatters the model.
              Judge it on the other two.
            </p>
          </div>
        </section>

        {/* 8 ------------------------------------------------------------- */}
        <section>
          <div className="card">
            <div className="eyebrow">Refuse to guess</div>
            <h2>An honest screening tool knows what it does not know.</h2>
            <p>
              Certus does not force a verdict on every eye. Two readers grade each photograph: Certus,
              which shows the lesions behind its grade, and a plain whole-image grader that shows nothing
              but grades a little more accurately. An eye is cleared or referred on its own only when both
              agree and both are sure. Anything else goes to a person with the evidence already attached,
              and the reason goes with it. An abstention is stored as an outcome, not as a blank field.
            </p>
            <p>
              What a health system needs to know is not how often the model is right. It is how many
              referable eyes the whole arrangement misses once human review is counted in. An accuracy
              figure without the abstention rate beside it is misleading, so the two always appear
              together.
            </p>
            <div className="grid">
              <div><dt>Test · referable caught</dt><dd>{M.two.test.sysSens}<small>sends {M.two.test.toHuman} to a person</small></dd></div>
              <div><dt>Unseen camera · caught</dt><dd className="teal">{M.two.ext.sysSens}<small>sends {M.two.ext.toHuman} to a person</small></dd></div>
            </div>
            <div className="note">
              <b>Where this stands today.</b> On a camera it has never seen, the pair catches every referable
              eye, but it hands {M.two.ext.toHuman} of eyes to a person to do it. That is nearly half, and too
              many. Re-fitting both readers on that camera's first {M.two.extRefit.patients} patients brings
              it down to {M.two.extRefit.toHuman}, still catching {M.two.extRefit.sysSens} of referable eyes at
              {" "}{M.two.extRefit.sysSpec} specificity. Until a camera has those patients, expect the higher number.
            </div>
          </div>
        </section>

        {/* 9 ------------------------------------------------------------- */}
        <section className="right">
          <div className="card">
            <div className="eyebrow">Camera-aware trust</div>
            <h2>Confidence is a property of the camera, not just the model.</h2>
            <p>
              A confidence correction fitted on one set of cameras does not carry over to another. We
              measured it. The same temperature cut calibration error on the internal test set from
              {" "}{M.band.test.eceRaw} to {M.band.test.ece}, and pushed it up on Messidor-2, from {M.ext.eceRaw} to
              {" "}{M.ext.ece}. Within the test set it ranged from {M.eceByCamera.aptos} to {M.eceByCamera.ddr}
              depending on the camera.
            </p>
            <p>
              So every camera family gets its own correction, fitted on held-out images from that
              family. A camera that was never part of calibration is marked unverified rather than
              quietly trusted, and the operator sees that on screen.
            </p>
            <div className="grid">
              <div><dt>Calibration error</dt><dd className="sienna">{M.ext.ece}<small>unseen camera, was {M.ext.eceRaw} before</small></dd></div>
              <div><dt>Unknown camera</dt><dd className="sienna">flagged<small>not silently graded</small></dd></div>
            </div>
          </div>
        </section>

        {/* 10 ------------------------------------------------------------ */}
        <section>
          <div className="card">
            <div className="eyebrow">The queue</div>
            <h2>One ophthalmologist, one district.</h2>
            <p>
              Accuracy only matters if it changes what a district can staff. We modelled one screening
              100,000 people a year across twenty camps, with power cuts, consent steps, reviewer
              working hours and the weekend gap all in the model. Then we asked how many
              ophthalmologists it takes to return 95% of results inside 48 hours.
            </p>
            <table className="data">
              <thead>
                <tr><th>Scenario</th><th style={{ textAlign: "right" }}>Manual</th><th style={{ textAlign: "right" }}>With Certus</th></tr>
              </thead>
              <tbody>
                <tr><td>100,000 patients / year</td><td className="n">3</td><td className="n">1</td></tr>
                <tr><td>Reviewer utilisation</td><td className="n">86%</td><td className="n">28%</td></tr>
                <tr><td>200,000 patients / year</td><td className="n">5</td><td className="n">1</td></tr>
                <tr className="hi"><td>Two review hours a day</td><td className="n">unreachable</td><td className="n">1</td></tr>
              </tbody>
            </table>
            <p className="small">
              The last row is the real constraint. Ophthalmologists run clinics too. Give them two
              hours a day for tele-review and manual grading cannot hit a 48-hour turnaround at any
              staffing level, because the queue outruns the weekend. Certus can.
            </p>
            <div className="src">
              Discrete-event model; rural supply 22.6 h/day (Ministry of Power, 2025). Review times
              and capture times are stated assumptions, swept across their plausible ranges.
            </div>
          </div>
        </section>

        {/* 11 ------------------------------------------------------------ */}
        <section className="right">
          <div className="card">
            <div className="eyebrow">What we have not done</div>
            <h2>This is a retrospective evaluation, not a clinical validation.</h2>
            <p>
              Certus has been evaluated on public datasets with leakage-free splits. Duplicate
              photographs are found by retinal texture and kept on the same side of every split, and
              Messidor-2 was held back completely so that "unseen camera" means what it says.
            </p>
            <div className="grid">
              <div><dt>Training eyes</dt><dd>{M.trainEyes}<small>four datasets</small></dd></div>
              <div><dt>Held out entirely</dt><dd>{M.ext.n}<small>Messidor-2</small></dd></div>
              <div><dt>Model</dt><dd style={{ fontSize: "0.95rem" }}>{M.model}</dd></div>
            </div>
            <p>
              It has not been tested on patients, in a camp, on the cameras it would actually meet.
              Prospective evaluation and a CDSCO Class C pathway come next. Nothing on this page
              should be read as a claim that either one is done.
            </p>
            <div className="legend">
              <span className="pill bush">Leakage-free splits</span>
              <span className="pill teal">External test set</span>
              <span className="pill oak">Reproducible: model + calibration versioned</span>
              <span className="pill sienna">Not yet clinically validated</span>
            </div>
          </div>
        </section>

        {/* 12 ------------------------------------------------------------ */}
        <section className="center">
          <div className="card narrow">
            <div className="eyebrow">In one line</div>
            <h2>Find the disease early, prove why, and send the doubtful cases to a person.</h2>
            <p className="dim">
              A screening tool earns its place by being checkable. Certus shows the evidence behind
              every grade, the confidence that grade has actually earned on that camera, and an audit
              trail nobody can quietly rewrite.
            </p>
          </div>
        </section>
      </main>
    </>
  );
}
