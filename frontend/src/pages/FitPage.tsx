import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button, Card, ErrorBanner, Field, FileInput, InfoPanel, PageHeader, SuccessBanner, Tabs, TextArea } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { TABULAR_ACCEPT, TABULAR_FORMATS_HINT } from "../lib/constants";
import { imageFilesToBase64, parseJointRecordsFile, parseTextSamplesFile } from "../lib/files";
import type { FitResponse } from "../lib/types";

type Modality = "tabular" | "text" | "image" | "joint";

// Checked against the real backend logic (db/crud.py, utils/profiler.py,
// adapters/text.py, adapters/image.py, adapters/joint.py) so this stays
// honest. Update it if that logic ever changes.
const LIMITATIONS: Record<Modality, { title: string; items: string[] }> = {
  tabular: {
    title: "A few things worth knowing before you upload",
    items: [
      "We'll clean up numeric columns for you (dropping anything that isn't a number, and telling you how many we dropped), but text categories aren't normalized. \"USA\", \"usa\", and \" USA \" will be treated as three separate categories.",
      "If a column is the same value all the way down, or just counts up/down in order (like a row number or an ID), we leave it out of monitoring automatically. That's intentional, not a bug, but it does mean that column just won't show up afterward.",
      "Categorical columns are capped at 50 distinct values. Go over that and the upload is rejected outright, since it's almost always a free-text or ID field that shouldn't be treated as categorical anyway.",
      "We don't deduplicate rows, strip outliers, or scrub personal data for you, so do that on your end first if it matters.",
    ],
  },
  text: {
    title: "A few things worth knowing before you upload",
    items: [
      "This wants plain text, already extracted. If you upload a PDF or Word doc, it'll just read the raw bytes, not the words inside, so the result will be meaningless.",
      "Anything past 256 tokens per sample gets cut off automatically by the embedding model. Long documents still work, they just get judged on their first chunk.",
      "There's no profanity filtering, PII scrubbing, or deduplication happening here.",
      "You need at least 4 samples for this to run at all, but aim for 40 or more. Fewer than that and the results get noisy.",
    ],
  },
  image: {
    title: "A few things worth knowing before you upload",
    items: [
      "Most common formats work fine (JPEG, PNG, and so on). A file that can't be read gets rejected with a clear error instead of silently vanishing.",
      "Every image gets resized and cropped to a 224×224 square before it's embedded. For anything that isn't roughly square to begin with, that crop can cut off real content, not just empty margin.",
      "No deduplication or quality checks happen automatically.",
      "Same as text: 4 images minimum to run, 40+ recommended for something you can actually trust.",
    ],
  },
  joint: {
    title: "A few things worth knowing before you upload",
    items: [
      "This is the newest of the four paths, and it hasn't gotten calibrated mode yet, so it only runs in legacy mode for now.",
      "Each record needs to already be shaped as tabular, text, and/or image fields. We can match attached images to records by filename, or just pair them up in order, but we won't figure out how your files relate to each other on our own.",
      "Everything above about text and images still applies here, for whichever part of a record you're including.",
    ],
  },
};

function ResultPanel({ result, onContinue }: { result: FitResponse; onContinue: () => void }) {
  return (
    <div className="space-y-4">
      <SuccessBanner message={result.message} />
      {result.calibration_info && Object.keys(result.calibration_info).length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3">Feature</th>
                <th className="px-5 py-3">Effect floor</th>
                <th className="px-5 py-3">Min. detectable D</th>
                <th className="px-5 py-3">Reference OK for floor?</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(result.calibration_info).map(([name, info]) => (
                <tr key={name}>
                  <td className="px-5 py-2.5 font-medium text-slate-700">{name}</td>
                  <td className="px-5 py-2.5 text-slate-600">{info.effect_floor ?? "--"}</td>
                  <td className="px-5 py-2.5 text-slate-600">
                    {info.minimum_detectable_d != null ? info.minimum_detectable_d.toFixed(4) : "--"}
                  </td>
                  <td className="px-5 py-2.5">
                    {info.reference_too_small_for_floor ? (
                      <span className="text-warn-600">Too small</span>
                    ) : (
                      <span className="text-ok-600">Yes</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      <Button onClick={onContinue}>Go to Analyze &rarr;</Button>
    </div>
  );
}

function SourceToggle({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="mb-4 flex gap-4 border-b border-slate-100 pb-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`-mb-px border-b-2 px-1 pb-2 text-sm font-medium transition-colors ${
            value === opt.value
              ? "border-brand-600 text-brand-700"
              : "border-transparent text-slate-400 hover:text-slate-600"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function FitPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [modality, setModality] = useState<Modality>("tabular");
  const [result, setResult] = useState<FitResponse | null>(null);

  // tabular
  const [tabularSource, setTabularSource] = useState<"file" | "paste">("file");
  const [tabularFile, setTabularFile] = useState<File | null>(null);
  const [pastedCsv, setPastedCsv] = useState("");

  // text
  const [textSource, setTextSource] = useState<"paste" | "file">("paste");
  const [referenceTexts, setReferenceTexts] = useState("");
  const [textFile, setTextFile] = useState<File | null>(null);
  const [textCalibrated, setTextCalibrated] = useState(false);

  // image
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imageCalibrated, setImageCalibrated] = useState(false);

  // joint
  const [jointSource, setJointSource] = useState<"file" | "json">("file");
  const [jointRecordsFile, setJointRecordsFile] = useState<File | null>(null);
  const [jointImageFiles, setJointImageFiles] = useState<File[]>([]);
  const [jointJson, setJointJson] = useState('[\n  {"tabular": {"amount": 42.5}, "text": "example note"}\n]');

  const onSuccess = (data: FitResponse) => {
    setResult(data);
    queryClient.invalidateQueries({ queryKey: ["projects"] });
    queryClient.invalidateQueries({ queryKey: ["baseline", projectId] });
  };

  const tabularMutation = useMutation({
    mutationFn: () => {
      if (tabularSource === "file") {
        if (!tabularFile) throw new Error("Choose a file first.");
        return api.fitUpload(projectId, tabularFile);
      }
      if (!pastedCsv.trim()) throw new Error("Paste some CSV data first.");
      const blob = new File([pastedCsv], "pasted.csv", { type: "text/csv" });
      return api.fitUpload(projectId, blob);
    },
    onSuccess,
  });

  const textMutation = useMutation({
    mutationFn: async () => {
      let texts: string[];
      if (textSource === "file") {
        if (!textFile) throw new Error("Choose a file first.");
        texts = await parseTextSamplesFile(textFile);
      } else {
        texts = referenceTexts.split("\n").map((t) => t.trim()).filter(Boolean);
      }
      return api.fitText(projectId, texts, textCalibrated ? { decision_mode: "calibrated" } : undefined);
    },
    onSuccess,
  });

  const imageMutation = useMutation({
    mutationFn: async () => {
      const b64 = await imageFilesToBase64(imageFiles);
      return api.fitImage(projectId, b64, imageCalibrated ? { decision_mode: "calibrated" } : undefined);
    },
    onSuccess,
  });

  const jointMutation = useMutation({
    mutationFn: async () => {
      if (jointSource === "file") {
        if (!jointRecordsFile) throw new Error("Choose a records file first.");
        const records = await parseJointRecordsFile(jointRecordsFile, jointImageFiles);
        return api.fitJoint(projectId, records);
      }
      const records = JSON.parse(jointJson);
      return api.fitJoint(projectId, records);
    },
    onSuccess,
  });

  const activeMutation = { tabular: tabularMutation, text: textMutation, image: imageMutation, joint: jointMutation }[modality];
  const error = activeMutation.error;

  return (
    <div>
      <PageHeader title="Fit Baseline" subtitle="Lock a reference distribution for this project." />

      <div className="mb-6">
        <Tabs
          value={modality}
          onChange={(m) => {
            setModality(m);
            setResult(null);
          }}
          options={[
            { value: "tabular", label: "Tabular" },
            { value: "text", label: "Text" },
            { value: "image", label: "Image" },
            { value: "joint", label: "Joint" },
          ]}
        />
      </div>

      {result ? (
        <ResultPanel result={result} onContinue={() => navigate(`/projects/${encodeURIComponent(projectId)}/analyze`)} />
      ) : (
        <div className="max-w-2xl space-y-4">
          <InfoPanel title={LIMITATIONS[modality].title} items={LIMITATIONS[modality].items} />
          <Card className="p-6">
          {modality === "tabular" && (
            <div className="space-y-4">
              <SourceToggle
                value={tabularSource}
                onChange={(v) => setTabularSource(v as "file" | "paste")}
                options={[
                  { value: "file", label: "Upload a file" },
                  { value: "paste", label: "Paste data" },
                ]}
              />
              {tabularSource === "file" ? (
                <Field label="Reference dataset" hint={TABULAR_FORMATS_HINT}>
                  <FileInput
                    accept={TABULAR_ACCEPT}
                    files={tabularFile ? [tabularFile] : []}
                    onFiles={(files) => setTabularFile(files[0] ?? null)}
                  />
                </Field>
              ) : (
                <Field label="Paste CSV data" hint="First row is treated as the header.">
                  <TextArea
                    rows={8}
                    value={pastedCsv}
                    onChange={(e) => setPastedCsv(e.target.value)}
                    placeholder={"amount,category\n42.5,retail\n13.0,food"}
                  />
                </Field>
              )}
              <Button
                onClick={() => tabularMutation.mutate()}
                disabled={tabularMutation.isPending || (tabularSource === "file" ? !tabularFile : !pastedCsv.trim())}
              >
                {tabularMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </div>
          )}

          {modality === "text" && (
            <div className="space-y-4">
              <SourceToggle
                value={textSource}
                onChange={(v) => setTextSource(v as "paste" | "file")}
                options={[
                  { value: "paste", label: "Paste text" },
                  { value: "file", label: "Upload a file" },
                ]}
              />
              {textSource === "paste" ? (
                <Field label="Reference texts" hint="One text sample per line. 40+ recommended.">
                  <TextArea rows={8} value={referenceTexts} onChange={(e) => setReferenceTexts(e.target.value)} />
                </Field>
              ) : (
                <Field label="Reference texts file" hint="A .txt file with one sample per line, or a .json/.jsonl file.">
                  <FileInput
                    accept=".txt,.json,.jsonl,.ndjson"
                    files={textFile ? [textFile] : []}
                    onFiles={(files) => setTextFile(files[0] ?? null)}
                  />
                </Field>
              )}
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={textCalibrated} onChange={(e) => setTextCalibrated(e.target.checked)} />
                Use calibrated mode (gives drift a real p-value instead of a flat cutoff)
              </label>
              <Button onClick={() => textMutation.mutate()} disabled={textMutation.isPending}>
                {textMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </div>
          )}

          {modality === "image" && (
            <div className="space-y-4">
              <Field label="Reference images" hint="JPEG, PNG, or most other common formats -- or a .zip of images. Pick a batch of 40 or more for a stable baseline.">
                <FileInput accept="image/*,.zip" multiple files={imageFiles} onFiles={setImageFiles} />
              </Field>
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={imageCalibrated} onChange={(e) => setImageCalibrated(e.target.checked)} />
                Use calibrated mode (gives drift a real p-value instead of a flat cutoff)
              </label>
              <Button onClick={() => imageMutation.mutate()} disabled={imageFiles.length === 0 || imageMutation.isPending}>
                {imageMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </div>
          )}

          {modality === "joint" && (
            <div className="space-y-4">
              <SourceToggle
                value={jointSource}
                onChange={(v) => setJointSource(v as "file" | "json")}
                options={[
                  { value: "file", label: "Upload files" },
                  { value: "json", label: "Write JSON" },
                ]}
              />
              {jointSource === "file" ? (
                <>
                  <Field
                    label="Records file"
                    hint='A .json or .jsonl file where each record looks like {"tabular": {...}, "text": "...", "image": "..."}. Any of the three can be left out. If an image is inline base64 already, it works as-is.'
                  >
                    <FileInput
                      accept=".json,.jsonl,.ndjson"
                      files={jointRecordsFile ? [jointRecordsFile] : []}
                      onFiles={(files) => setJointRecordsFile(files[0] ?? null)}
                    />
                  </Field>
                  <Field
                    label="Attach images (optional)"
                    hint="Got images as separate files instead of inline base64? Attach them here and we'll match them to your records by filename, or just pair them up in order if there's nothing to match on."
                  >
                    <FileInput accept="image/*" multiple files={jointImageFiles} onFiles={setJointImageFiles} />
                  </Field>
                </>
              ) : (
                <Field label="Reference records (JSON)" hint='Array of {"tabular"?, "text"?, "image"?} records.'>
                  <TextArea rows={10} value={jointJson} onChange={(e) => setJointJson(e.target.value)} />
                </Field>
              )}
              <Button
                onClick={() => jointMutation.mutate()}
                disabled={jointMutation.isPending || (jointSource === "file" && !jointRecordsFile)}
              >
                {jointMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </div>
          )}

          {error && (
            <div className="mt-4">
              <ErrorBanner
                message={error instanceof ApiError ? error.detail : error instanceof Error ? error.message : "Fit failed."}
              />
            </div>
          )}
          </Card>
        </div>
      )}
    </div>
  );
}
