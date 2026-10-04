import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button, Card, ErrorBanner, Field, PageHeader, SuccessBanner, Tabs, TextArea } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { filesToBase64 } from "../lib/files";
import type { FitResponse } from "../lib/types";

type Modality = "tabular" | "text" | "image" | "joint";

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

export function FitPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [modality, setModality] = useState<Modality>("tabular");
  const [result, setResult] = useState<FitResponse | null>(null);

  // tabular
  const [tabularFile, setTabularFile] = useState<File | null>(null);
  // text
  const [referenceTexts, setReferenceTexts] = useState("");
  const [textCalibrated, setTextCalibrated] = useState(false);
  // image
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imageCalibrated, setImageCalibrated] = useState(false);
  // joint
  const [jointJson, setJointJson] = useState('[\n  {"tabular": {"amount": 42.5}, "text": "example note"}\n]');

  const onSuccess = (data: FitResponse) => {
    setResult(data);
    queryClient.invalidateQueries({ queryKey: ["projects"] });
    queryClient.invalidateQueries({ queryKey: ["baseline", projectId] });
  };

  const tabularMutation = useMutation({
    mutationFn: () => {
      if (!tabularFile) throw new Error("Choose a file first.");
      return api.fitUpload(projectId, tabularFile);
    },
    onSuccess,
  });

  const textMutation = useMutation({
    mutationFn: () => {
      const texts = referenceTexts.split("\n").map((t) => t.trim()).filter(Boolean);
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
    mutationFn: () => {
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
        <Card className="max-w-2xl p-6 space-y-4">
          {modality === "tabular" && (
            <>
              <Field label="Reference dataset" hint="CSV or Parquet, one row per observation.">
                <input
                  type="file"
                  accept=".csv,.parquet"
                  onChange={(e) => setTabularFile(e.target.files?.[0] ?? null)}
                  className="block w-full text-sm text-slate-600"
                />
              </Field>
              <Button onClick={() => tabularMutation.mutate()} disabled={!tabularFile || tabularMutation.isPending}>
                {tabularMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </>
          )}

          {modality === "text" && (
            <>
              <Field label="Reference texts" hint="One text sample per line. 40+ recommended.">
                <TextArea rows={8} value={referenceTexts} onChange={(e) => setReferenceTexts(e.target.value)} />
              </Field>
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={textCalibrated} onChange={(e) => setTextCalibrated(e.target.checked)} />
                Use calibrated decision mode (p-value via precomputed null grid, opt-in)
              </label>
              <Button onClick={() => textMutation.mutate()} disabled={textMutation.isPending}>
                {textMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </>
          )}

          {modality === "image" && (
            <>
              <Field label="Reference images" hint="40+ images recommended.">
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
            </>
          )}

          {modality === "joint" && (
            <>
              <Field label="Reference records (JSON)" hint='Array of {"tabular"?, "text"?, "image"?} records.'>
                <TextArea rows={10} value={jointJson} onChange={(e) => setJointJson(e.target.value)} />
              </Field>
              <Button onClick={() => jointMutation.mutate()} disabled={jointMutation.isPending}>
                {jointMutation.isPending ? "Fitting..." : "Lock Baseline"}
              </Button>
            </>
          )}

          {error && (
            <ErrorBanner
              message={error instanceof ApiError ? error.detail : error instanceof Error ? error.message : "Fit failed."}
            />
          )}
        </Card>
      )}
    </div>
  );
}
