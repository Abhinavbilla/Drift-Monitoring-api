import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { AnalyzeResultPanel } from "../components/AnalyzeResultPanel";
import { Button, Card, ErrorBanner, Field, FileInput, PageHeader, Tabs, TextArea, TextInput } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { TABULAR_ACCEPT, TABULAR_FORMATS_HINT } from "../lib/constants";
import { imageFilesToBase64 } from "../lib/files";
import type { AnalyzeResponse } from "../lib/types";

type Modality = "tabular" | "text" | "image" | "joint";

export function AnalyzePage() {
  const { projectId = "" } = useParams();
  const [modality, setModality] = useState<Modality>("tabular");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const [tabularFile, setTabularFile] = useState<File | null>(null);
  const [idempotencyKey, setIdempotencyKey] = useState("");
  const [baselineVersion, setBaselineVersion] = useState<string>("");
  const [productionTexts, setProductionTexts] = useState("");
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [jointJson, setJointJson] = useState('[\n  {"tabular": {"amount": 99.0}, "text": "example note"}\n]');

  const { data: versionsData } = useQuery({
    queryKey: ["baselineVersions", projectId],
    queryFn: () => api.listBaselineVersions(projectId),
    retry: false,
  });

  const onSuccess = (data: AnalyzeResponse) => setResult(data);

  const tabularMutation = useMutation({
    mutationFn: () => {
      if (!tabularFile) throw new Error("Choose a file first.");
      return api.analyzeUpload(projectId, tabularFile, idempotencyKey || undefined);
    },
    onSuccess,
  });

  const textMutation = useMutation({
    mutationFn: () => {
      const texts = productionTexts.split("\n").map((t) => t.trim()).filter(Boolean);
      return api.analyzeText(projectId, texts);
    },
    onSuccess,
  });

  const imageMutation = useMutation({
    mutationFn: async () => {
      const b64 = await imageFilesToBase64(imageFiles);
      return api.analyzeImage(projectId, b64);
    },
    onSuccess,
  });

  const jointMutation = useMutation({
    mutationFn: () => api.analyzeJoint(projectId, JSON.parse(jointJson)),
    onSuccess,
  });

  const activeMutation = { tabular: tabularMutation, text: textMutation, image: imageMutation, joint: jointMutation }[modality];
  const error = activeMutation.error;

  return (
    <div>
      <PageHeader title="Analyze" subtitle="Compare a production batch against the locked baseline." />

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

      <Card className="max-w-2xl p-6 space-y-4 mb-6">
        {modality === "tabular" && (
          <>
            <Field label="Production batch" hint={TABULAR_FORMATS_HINT}>
              <FileInput
                accept={TABULAR_ACCEPT}
                files={tabularFile ? [tabularFile] : []}
                onFiles={(files) => setTabularFile(files[0] ?? null)}
              />
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Idempotency-Key (optional)">
                <TextInput value={idempotencyKey} onChange={(e) => setIdempotencyKey(e.target.value)} placeholder="batch-2026-10-04" />
              </Field>
              <Field label="Baseline version (optional)">
                <select
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                  value={baselineVersion}
                  onChange={(e) => setBaselineVersion(e.target.value)}
                >
                  <option value="">Active</option>
                  {versionsData?.versions.map((v) => (
                    <option key={v.version} value={v.version}>
                      v{v.version} {v.active ? "(active)" : ""}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Button onClick={() => tabularMutation.mutate()} disabled={!tabularFile || tabularMutation.isPending}>
              {tabularMutation.isPending ? "Analyzing..." : "Run Analysis"}
            </Button>
          </>
        )}

        {modality === "text" && (
          <>
            <Field label="Production texts" hint="One text sample per line.">
              <TextArea rows={8} value={productionTexts} onChange={(e) => setProductionTexts(e.target.value)} />
            </Field>
            <Button onClick={() => textMutation.mutate()} disabled={textMutation.isPending}>
              {textMutation.isPending ? "Analyzing..." : "Run Analysis"}
            </Button>
          </>
        )}

        {modality === "image" && (
          <>
            <Field label="Production images" hint="Image files, or a .zip of images.">
              <FileInput accept="image/*,.zip" multiple files={imageFiles} onFiles={setImageFiles} />
            </Field>
            <Button onClick={() => imageMutation.mutate()} disabled={imageFiles.length === 0 || imageMutation.isPending}>
              {imageMutation.isPending ? "Analyzing..." : "Run Analysis"}
            </Button>
          </>
        )}

        {modality === "joint" && (
          <>
            <Field label="Production records (JSON)">
              <TextArea rows={10} value={jointJson} onChange={(e) => setJointJson(e.target.value)} />
            </Field>
            <Button onClick={() => jointMutation.mutate()} disabled={jointMutation.isPending}>
              {jointMutation.isPending ? "Analyzing..." : "Run Analysis"}
            </Button>
          </>
        )}

        {error && (
          <ErrorBanner
            message={error instanceof ApiError ? error.detail : error instanceof Error ? error.message : "Analysis failed."}
          />
        )}
      </Card>

      {result && <AnalyzeResultPanel result={result} />}
    </div>
  );
}
