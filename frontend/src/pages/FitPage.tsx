import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button, Card, ErrorBanner, Field, InfoPanel, PageHeader, SuccessBanner, Tabs, TextArea } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { TABULAR_ACCEPT, TABULAR_FORMATS_HINT } from "../lib/constants";
import { filesToBase64, parseJointRecordsFile, parseTextSamplesFile } from "../lib/files";
import type { FitResponse } from "../lib/types";

type Modality = "tabular" | "text" | "image" | "joint";

// Verified against the actual backend code (db/crud.py, utils/profiler.py,
// adapters/text.py, adapters/image.py, adapters/joint.py) -- not assumed.
// Keep this in sync if any of that logic changes.
const LIMITATIONS: Record<Modality, { title: string; items: string[] }> = {
  tabular: {
    title: "Before you upload a tabular dataset",
    items: [
      "Cleaning is narrow: a numeric column has non-numeric cells automatically dropped (and reported), but categorical values are NOT normalized -- \"USA\", \"usa\", and \" USA \" are counted as three different categories, not merged.",
      "A column that's constant, or a perfectly increasing/decreasing sequence (e.g. a row index or auto-incrementing ID), is silently excluded from monitoring entirely -- by design, not an error, but it means that column won't show up at all afterward.",
      "A categorical column with more than 50 unique values is rejected outright as likely a free-text or ID field -- it is not truncated, sampled down, or auto-converted to continuous.",
      "No deduplication, outlier removal, or PII redaction happens automatically -- if your raw export has duplicate rows or sensitive fields, remove them yourself first.",
    ],
  },
  text: {
    title: "Before you upload text",
    items: [
      "This expects already-extracted plain text strings. It does NOT extract text from PDFs, Word docs, or HTML -- uploading one of those embeds the raw file bytes as garbage, not its readable content.",
      "Long text is silently truncated to the embedding model's 256-token input limit -- anything beyond that is ignored when comparing batches.",
      "No profanity/PII scrubbing, deduplication, or language filtering happens automatically.",
      "Needs at least 4 samples to run at all; 40+ is recommended for a stable signal (fewer than that, results get noisy).",
    ],
  },
  image: {
    title: "Before you upload images",
    items: [
      "Standard formats PIL can decode work (JPEG, PNG, and similar) -- a corrupted or unsupported file is rejected cleanly with an error, not silently skipped.",
      "Every image is resized and center-cropped to 224×224 before embedding -- for a very wide, tall, or otherwise non-square image, this crops out real content, not just margins.",
      "No deduplication or quality filtering happens automatically.",
      "Needs at least 4 images to run at all; 40+ is recommended for a stable signal.",
    ],
  },
  joint: {
    title: "Before you upload joint (multimodal) data",
    items: [
      "The least mature of the four paths -- it does not yet support calibrated decision mode (p-values), only the legacy AUC cutoff.",
      "Each record needs to already be organized as {tabular?, text?, image?} -- this page can match attached images to records by filename or by order, but it won't auto-detect or auto-pair files on its own.",
      "Every limitation listed above for text and images applies here too, for whichever part of a record is present.",
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
      const b64 = await filesToBase64(imageFiles);
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
                  <input
                    type="file"
                    accept={TABULAR_ACCEPT}
                    onChange={(e) => setTabularFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-sm text-slate-600"
                  />
                  {tabularFile && <p className="mt-1 text-xs text-slate-400">{tabularFile.name}</p>}
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
                <Field label="Reference texts file" hint=".txt (one per line), .json (array of strings), or .jsonl/.ndjson.">
                  <input
                    type="file"
                    accept=".txt,.json,.jsonl,.ndjson"
                    onChange={(e) => setTextFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-sm text-slate-600"
                  />
                  {textFile && <p className="mt-1 text-xs text-slate-400">{textFile.name}</p>}
                </Field>
              )}
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={textCalibrated} onChange={(e) => setTextCalibrated(e.target.checked)} />
                Use calibrated decision mode (p-value via precomputed null grid, opt-in)
              </label>
              <Button onClick={() => textMutation.mutate()} disabled={textMutation.isPending}>
                {textMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </div>
          )}

          {modality === "image" && (
            <div className="space-y-4">
              <Field label="Reference images" hint="PNG, JPEG, or any browser-readable image format. 40+ recommended.">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => setImageFiles(Array.from(e.target.files ?? []))}
                  className="block w-full text-sm text-slate-600"
                />
              </Field>
              {imageFiles.length > 0 && <p className="text-xs text-slate-400">{imageFiles.length} file(s) selected</p>}
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={imageCalibrated} onChange={(e) => setImageCalibrated(e.target.checked)} />
                Use calibrated decision mode (p-value via precomputed null grid, opt-in)
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
                    hint='.json (array of records) or .jsonl. Each record: {"tabular"?, "text"?, "image"?}. An image field can be inline base64, or a filename matching one of the files attached below.'
                  >
                    <input
                      type="file"
                      accept=".json,.jsonl,.ndjson"
                      onChange={(e) => setJointRecordsFile(e.target.files?.[0] ?? null)}
                      className="block w-full text-sm text-slate-600"
                    />
                    {jointRecordsFile && <p className="mt-1 text-xs text-slate-400">{jointRecordsFile.name}</p>}
                  </Field>
                  <Field
                    label="Attach images (optional)"
                    hint="If your records file doesn't already embed images inline, attach them here -- matched by filename, or assigned in order to records without an image."
                  >
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={(e) => setJointImageFiles(Array.from(e.target.files ?? []))}
                      className="block w-full text-sm text-slate-600"
                    />
                    {jointImageFiles.length > 0 && (
                      <p className="mt-1 text-xs text-slate-400">{jointImageFiles.length} image(s) attached</p>
                    )}
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
