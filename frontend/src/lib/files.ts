export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // strip the "data:<mime>;base64," prefix -- the backend expects raw base64
      resolve(result.split(",")[1] ?? "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function filesToBase64(files: File[]): Promise<string[]> {
  return Promise.all(files.map(fileToBase64));
}

export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

/** Parses an uploaded .txt (one sample per line), .json (array of
 * strings), or .jsonl/.ndjson (one JSON string per line) file into a
 * flat string array -- for the text modality's "upload a file I
 * already have" option, instead of requiring manual paste. */
export async function parseTextSamplesFile(file: File): Promise<string[]> {
  const content = await readFileAsText(file);
  const name = file.name.toLowerCase();
  if (name.endsWith(".json")) {
    const parsed = JSON.parse(content);
    if (!Array.isArray(parsed)) throw new Error(".json file must contain a JSON array of strings.");
    return parsed.map((v) => String(v));
  }
  if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) {
    return content.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
      try {
        const parsed = JSON.parse(l);
        return typeof parsed === "string" ? parsed : JSON.stringify(parsed);
      } catch {
        return l;
      }
    });
  }
  // plain .txt (or anything else) -- one sample per line
  return content.split("\n").map((l) => l.trim()).filter(Boolean);
}

interface ParsedJointRecord {
  tabular?: Record<string, unknown>;
  text?: string;
  image?: string;
  [key: string]: unknown;
}

/** Parses an uploaded .json (array of records) or .jsonl/.ndjson (one
 * record per line) joint-records file. Each record may already carry
 * an inline base64 `image`, OR reference one by filename (matched
 * against attachedImages below) -- supports a user who has "a file
 * with both text and images" either as one fully-inline file, or as a
 * metadata file plus a separate folder of image files. */
export async function parseJointRecordsFile(
  file: File, attachedImages: File[] = [],
): Promise<ParsedJointRecord[]> {
  const content = await readFileAsText(file);
  const name = file.name.toLowerCase();
  let records: ParsedJointRecord[];
  if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) {
    records = content.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));
  } else {
    const parsed = JSON.parse(content);
    if (!Array.isArray(parsed)) throw new Error("Joint records file must contain a JSON array (or use .jsonl).");
    records = parsed;
  }

  if (attachedImages.length === 0) return records;

  const imagesByName = new Map(attachedImages.map((f) => [f.name, f]));
  const base64ByName = new Map<string, string>();
  for (const f of attachedImages) {
    base64ByName.set(f.name, await fileToBase64(f));
  }

  let nextUnassigned = 0;
  return records.map((record) => {
    if (record.image && typeof record.image === "string" && imagesByName.has(record.image)) {
      return { ...record, image: base64ByName.get(record.image) };
    }
    if (!record.image && nextUnassigned < attachedImages.length) {
      const b64 = base64ByName.get(attachedImages[nextUnassigned].name);
      nextUnassigned += 1;
      return { ...record, image: b64 };
    }
    return record;
  });
}
