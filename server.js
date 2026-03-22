// server.js — Solfai v5: All bugs fixed from v4 audit
// FIXES:
// BUG1: mediaResolution value was lowercase snake_case → now correct uppercase enum
// BUG2: thinking_budget was camelCase in REST body → now snake_case
// BUG3: responseSchema/responseMimeType were camelCase in REST body → now snake_case
// BUG4: C# major and A# minor missing from key enum → added
// BUG5: first_notes now used to pre-populate solfege preview
// BUG6: difficulty_text was mislabeled as "intervals" → fixed schema + mapping
// BUG7: Solfege staff ID now sends all pages, not just page 1
// BUG8: Solfege images now preprocessed same as analyze
// BUG9: PDFs now send BOTH raw PDF + JPEG pages for maximum accuracy
// BUG10: Retry logic added for Gemini API (3 attempts, exponential backoff)
// BUG11: Key disagreement warning flag added to response
// BUG12: mediaResolution fixed in chat handler too

import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createHash } from 'crypto';
import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'fs';
import sharp from 'sharp';
import multer from 'multer';
import AdmZip from 'adm-zip';

const upload = multer({ dest: '/tmp/solfai-uploads/' });

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '25mb' }));
app.use(express.static(join(__dirname, 'public')));

// ─── Config ───────────────────────────────────────────────
const GEMINI_MODEL = 'gemini-2.5-pro';
const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const CORRECTIONS_FILE = join(__dirname, 'corrections.json');

// ─── Correction Cache ─────────────────────────────────────
function loadCorrections() {
  try {
    if (existsSync(CORRECTIONS_FILE)) return JSON.parse(readFileSync(CORRECTIONS_FILE, 'utf8'));
  } catch (_) {}
  return {};
}
function saveCorrections(data) {
  try { writeFileSync(CORRECTIONS_FILE, JSON.stringify(data, null, 2)); } catch (_) {}
}
function hashImage(base64Data) {
  const chunk = (base64Data || '').substring(0, 50000);
  return createHash('md5').update(chunk).digest('hex');
}

// ─── responseSchema with enum constraints ─────────────────
// FIX BUG4: Added C# major and A# minor which were missing
const ANALYZE_SCHEMA = {
  type: "OBJECT",
  properties: {
    key_signature: {
      type: "STRING",
      description: "The key signature. Count accidentals carefully between the clef and time signature. Pick the closest match from the list.",
      enum: [
        "C major","G major","D major","A major","E major","B major","F# major","C# major","Cb major",
        "F major","Bb major","Eb major","Ab major","Db major","Gb major",
        "A minor","E minor","B minor","F# minor","C# minor","G# minor","D# minor","A# minor",
        "D minor","G minor","C minor","F minor","Bb minor","Eb minor","Ab minor"
      ]
    },
    time_signature: {
      type: "STRING",
      description: "The time signature at the beginning of the piece.",
      enum: ["4/4","3/4","2/4","4/8","6/8","9/8","12/8","2/2","3/8","3/2","6/4","5/4","7/8"]
    },
    tempo: {
      type: "STRING",
      description: "The tempo marking written on the score (e.g., 'Andante', 'Allegro q=120'). Write 'none' if not visible.",
    },
    starting_pitch: {
      type: "STRING",
      description: "The first note SUNG by the vocal part (the staff with lyrics). Skip piano introductions. Find where lyrics begin. For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.",
      enum: [
        "C3","C#3","D3","Eb3","E3","F3","F#3","G3","Ab3","A3","Bb3","B3",
        "C4","C#4","D4","Eb4","E4","F4","F#4","G4","Ab4","A4","Bb4","B4",
        "C5","C#5","D5","Eb5","E5","F5","F#5","G5","Ab5","A5","Bb5","B5",
        "C6","D6","E6","F6","G6"
      ]
    },
    dynamics: {
      type: "STRING",
      description: "Opening dynamic marking (e.g., 'mp', 'f', 'pp') and any subsequent changes with measure numbers. Write 'none' if not visible."
    },
    flat_count: {
      type: "INTEGER",
      description: "Number of flats in the key signature. 0 if no flats. Count carefully — each flat symbol (♭) counts as 1."
    },
    sharp_count: {
      type: "INTEGER",
      description: "Number of sharps in the key signature. 0 if no sharps. Count carefully — each sharp symbol (♯) counts as 1."
    },
    // FIX BUG5: first_notes now used for solfege preview
    first_notes: {
      type: "ARRAY",
      items: { type: "STRING" },
      description: "First 12 note letter names with octave for the selected vocal part only (e.g., ['C4','E4','G4']). Include accidentals: 'Bb4', 'F#4'."
    },
    first_lyrics: {
      type: "STRING",
      description: "The first line of lyrics visible under the vocal part, copied exactly as written."
    },
    piece_title: {
      type: "STRING",
      description: "Title of the piece if visible on the score. Write 'unknown' if not visible."
    },
    composer_name: {
      type: "STRING",
      description: "Composer name if visible on the score. Write 'unknown' if not visible."
    },
    lyrics_language: {
      type: "STRING",
      description: "Language of the lyrics.",
      enum: ["English","Latin","German","French","Italian","Spanish","Hebrew","Russian","Other"]
    },
    // FIX BUG6: Separated difficulty_pitch and difficulty_intervals (were conflated)
    difficulty_overall: { type: "INTEGER", description: "Overall difficulty 1-10 for a community choir singer." },
    difficulty_rhythm: { type: "INTEGER", description: "Rhythm complexity 1-10. Consider syncopation, mixed meters, complex subdivisions." },
    difficulty_pitch: { type: "INTEGER", description: "Pitch range difficulty 1-10. Consider the range required and whether it is comfortable for the voice part." },
    difficulty_intervals: { type: "INTEGER", description: "Interval difficulty 1-10. Consider leaps, chromaticism, and whether the part is easy to tune." },
    difficulty_text: { type: "INTEGER", description: "Text/language difficulty 1-10. Higher if foreign language or complex diction." },
  },
  required: ["key_signature","time_signature","starting_pitch","dynamics","flat_count","sharp_count","difficulty_overall"]
};

// ─── Key from flat/sharp count (code, not AI) ─────────────
const KEY_FROM_COUNT = {
  '0':  { major: 'C major', minor: 'A minor' },
  '1b': { major: 'F major', minor: 'D minor' },
  '2b': { major: 'Bb major', minor: 'G minor' },
  '3b': { major: 'Eb major', minor: 'C minor' },
  '4b': { major: 'Ab major', minor: 'F minor' },
  '5b': { major: 'Db major', minor: 'Bb minor' },
  '6b': { major: 'Gb major', minor: 'Eb minor' },
  '7b': { major: 'Cb major', minor: 'Ab minor' },
  '1s': { major: 'G major', minor: 'E minor' },
  '2s': { major: 'D major', minor: 'B minor' },
  '3s': { major: 'A major', minor: 'F# minor' },
  '4s': { major: 'E major', minor: 'C# minor' },
  '5s': { major: 'B major', minor: 'G# minor' },
  '6s': { major: 'F# major', minor: 'D# minor' },
  '7s': { major: 'C# major', minor: 'A# minor' },
};

function resolveKeyFromCounts(flatCount, sharpCount, geminiKey) {
  let code;
  if (sharpCount > 0) code = `${sharpCount}s`;
  else if (flatCount > 0) code = `${flatCount}b`;
  else code = '0';

  const entry = KEY_FROM_COUNT[code];
  if (!entry) return { key: geminiKey || 'Unknown', confident: false };

  // MAJOR BIAS FIX: Default to major — most choral music (especially spirituals, folk songs,
  // classical SATB repertoire) is in major. Only use minor if Gemini says minor AND the exact
  // minor key it named matches what we'd expect from the accidental count.
  const geminiSaysMinor = (geminiKey || '').toLowerCase().includes('minor');
  const geminiMatchesExpectedMinor = geminiSaysMinor &&
    (geminiKey || '').toLowerCase().replace(/\s+/g,'') === entry.minor.toLowerCase().replace(/\s+/g,'');

  // Prefer major unless gemini is explicit AND correct about the minor key
  const keyName = geminiMatchesExpectedMinor ? entry.minor : entry.major;

  const accLabel = sharpCount > 0 ? `${sharpCount} sharp${sharpCount > 1 ? 's' : ''}` :
                   flatCount > 0 ? `${flatCount} flat${flatCount > 1 ? 's' : ''}` : 'no sharps or flats';

  const confident = geminiKey && (
    geminiKey.toLowerCase().replace(/\s+/g,'') === keyName.toLowerCase().replace(/\s+/g,'')
  );

  return {
    key: `${keyName} (${accLabel})`,
    confident: !!confident,
    geminiSaid: geminiKey,
    codeSaid: keyName,
  };
}

// ─── Solfege from note names (code, not AI) ───────────────
const NOTE_TO_SEMI = { 'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11 };
function noteToSolfege(noteName, tonicName) {
  if (!noteName || !tonicName) return noteName;
  const getSemi = (n) => {
    const letter = n[0].toUpperCase();
    let s = NOTE_TO_SEMI[letter];
    if (s == null) return null;
    for (let i = 1; i < n.length; i++) {
      if (n[i] === '#' || n[i] === '♯') s++;
      if (n[i] === 'b' || n[i] === '♭') s--;
    }
    return ((s % 12) + 12) % 12;
  };
  const ts = getSemi(tonicName), ns = getSemi(noteName);
  if (ts == null || ns == null) return noteName;
  const interval = ((ns - ts) % 12 + 12) % 12;
  const map = {0:'Do',2:'Re',4:'Mi',5:'Fa',7:'Sol',9:'La',11:'Ti',1:'Di',3:'Me',6:'Fi',8:'Si',10:'Te'};
  return map[interval] || noteName;
}

// ─── Image preprocessing with sharp ───────────────────────
async function preprocessForGemini(base64Data, mode = 'full') {
  try {
    const buf = Buffer.from(base64Data, 'base64');
    let pipeline;

    if (mode === 'key_region') {
      const meta = await sharp(buf).metadata();
      pipeline = sharp(buf)
        .extract({
          left: 0, top: 0,
          width: Math.floor(meta.width * 0.35),
          height: Math.floor(meta.height * 0.22)
        })
        .resize({ width: 1200 })
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.0 })
        .threshold(180)
        .jpeg({ quality: 95 });
    } else if (mode === 'binarize') {
      // High-contrast binarized mode for solfege extraction — pure B/W
      pipeline = sharp(buf)
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.0 })
        .threshold(160)
        .jpeg({ quality: 95 });
    } else {
      pipeline = sharp(buf)
        .grayscale()
        .normalise()
        .sharpen({ sigma: 1.5 })
        .jpeg({ quality: 92 });
    }

    const processed = await pipeline.toBuffer();
    return processed.toString('base64');
  } catch (e) {
    console.error('Preprocessing failed, using original:', e.message);
    return base64Data;
  }
}

// ─── Image segmentation for tall images ──────────────────
async function segmentImage(base64Data, maxSegments = 3) {
  try {
    const buf = Buffer.from(base64Data, 'base64');
    const meta = await sharp(buf).metadata();

    // Only segment if image is tall (aspect ratio suggests multiple systems)
    // Typical single-line sheet music is wider than tall
    if (meta.height < meta.width * 0.8 || meta.height < 800) {
      return null; // Not tall enough to need segmentation
    }

    const numSegments = Math.min(maxSegments, Math.ceil(meta.height / (meta.width * 0.4)));
    if (numSegments <= 1) return null;

    const segmentHeight = Math.ceil(meta.height / numSegments);
    const segments = [];

    for (let i = 0; i < numSegments; i++) {
      const top = i * segmentHeight;
      const height = Math.min(segmentHeight, meta.height - top);
      if (height < 100) break;

      const segBuf = await sharp(buf)
        .extract({ left: 0, top, width: meta.width, height })
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.0 })
        .threshold(160)
        .jpeg({ quality: 95 })
        .toBuffer();

      segments.push(segBuf.toString('base64'));
    }

    return segments.length > 1 ? segments : null;
  } catch (e) {
    console.error('Segmentation failed:', e.message);
    return null;
  }
}

// ─── Red box annotation for key signature ─────────────────
async function annotateKeyRegion(base64Data) {
  try {
    const buf = Buffer.from(base64Data, 'base64');
    const meta = await sharp(buf).metadata();
    const boxW = Math.floor(meta.width * 0.30);
    const boxH = Math.floor(meta.height * 0.20);

    const svg = `<svg width="${meta.width}" height="${meta.height}">
      <rect x="4" y="4" width="${boxW}" height="${boxH}" fill="none" stroke="red" stroke-width="8"/>
      <text x="12" y="${boxH + 35}" font-size="32" fill="red" font-weight="bold">KEY SIGNATURE REGION</text>
    </svg>`;

    const annotated = await sharp(buf)
      .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
      .jpeg({ quality: 92 })
      .toBuffer();

    return annotated.toString('base64');
  } catch (e) {
    console.error('Annotation failed:', e.message);
    return base64Data;
  }
}

// ─── Image builder ────────────────────────────────────────
// FIX BUG9: For PDFs, send BOTH raw PDF (text layer) AND JPEG pages (visual)
function buildImageParts(imageBase64, imageMime, pdfPages) {
  const parts = [];

  if (imageMime === 'application/pdf' && imageBase64) {
    // Send raw PDF first (Gemini reads native text layer for lyrics, composer, tempo)
    parts.push({ inlineData: { mimeType: 'application/pdf', data: imageBase64 } });
    // ALSO send JPEG renders for visual analysis (notes, accidentals, clefs)
    // This gives Gemini both the text layer AND pixel-level visual detail
    if (pdfPages?.length > 0) {
      for (const page of pdfPages.slice(0, 3)) {
        parts.push({ inlineData: { mimeType: 'image/jpeg', data: page } });
      }
    }
    return parts;
  }

  if (pdfPages?.length > 0) {
    for (const page of pdfPages) {
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: page } });
    }
  } else if (imageBase64) {
    parts.push({ inlineData: { mimeType: imageMime || 'image/jpeg', data: imageBase64 } });
  }
  return parts;
}

// ─── Gemini caller with retry ─────────────────────────────
// FIX BUG1: mediaResolution value is now correct uppercase enum string
// FIX BUG2: thinking_budget is now snake_case in REST body
// FIX BUG3: response_schema and response_mime_type are now snake_case
// FIX BUG15: Added retry logic with exponential backoff
async function callGemini(apiKey, systemPrompt, userParts, opts = {}) {
  const {
    temperature = 0,
    maxOutputTokens = 16384,
    model = GEMINI_MODEL,
    responseSchema = null,
    tools = null,
    thinkingBudget = 8000,
  } = opts;

  const url = `${BASE_URL}/${model}:generateContent?key=${apiKey}`;

  // FIX BUG1 + BUG2 + BUG3: All field names must be snake_case for the REST API
  const genConfig = {
    temperature,
    max_output_tokens: maxOutputTokens,
    thinking_config: { thinking_budget: thinkingBudget },
    media_resolution: 'MEDIA_RESOLUTION_HIGH',  // FIX BUG1: was 'media_resolution_high' (wrong)
  };

  if (responseSchema) {
    genConfig.response_mime_type = 'application/json';  // FIX BUG3: was responseMimeType
    genConfig.response_schema = responseSchema;          // FIX BUG3: was responseSchema
  }

  const body = {
    contents: [{ role: 'user', parts: userParts }],
    system_instruction: { parts: [{ text: systemPrompt }] },
    generation_config: genConfig,
  };

  if (tools) body.tools = tools;

  // FIX BUG15: Retry with exponential backoff
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const errText = await resp.text();
        let detail = `Gemini error: ${resp.status}`;
        try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}

        // Retry on rate limit (429) or server error (5xx)
        if ((resp.status === 429 || resp.status >= 500) && attempt < maxAttempts) {
          const delay = attempt * 2000; // 2s, 4s
          console.warn(`[Solfai] Gemini ${resp.status}, retrying in ${delay}ms (attempt ${attempt}/${maxAttempts})`);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        throw new Error(detail);
      }

      const data = await resp.json();
      const candidate = data.candidates?.[0];
      if (!candidate?.content?.parts) throw new Error('No response from Gemini');

      return candidate.content.parts
        .filter(p => p.text && !p.thought)
        .map(p => p.text)
        .join('')
        .trim();

    } catch (err) {
      if (attempt === maxAttempts) throw err;
      const delay = attempt * 2000;
      console.warn(`[Solfai] Network error, retrying in ${delay}ms:`, err.message);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

// ─── API Route ────────────────────────────────────────────
app.post('/api/analyze', async (req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'GEMINI_API_KEY not configured' });

  try {
    const { messages, imageBase64, imageMime, pdfPages, mode, selectedPart } = req.body;
    const part = selectedPart || 'Soprano';
    const imageParts = buildImageParts(imageBase64, imageMime, pdfPages);
    if (!imageParts.length) return res.status(400).json({ error: 'No image provided' });

    console.log(`[Solfai v7] mode=${mode}, part=${part}, parts=${imageParts.length}, mime=${imageMime || 'jpeg'}`);

    switch (mode) {
      case 'analyze': return await handleAnalyze(res, apiKey, imageParts, part, imageBase64, pdfPages);
      case 'solfege': return await handleSolfege(res, apiKey, imageParts, part);
      case 'rhythm':  return await handleRhythm(res, apiKey, imageParts, part);
      case 'chat':    return await handleChat(res, apiKey, messages, imageParts, part);
      case 'correct': return handleCorrection(res, req.body);
      default:        return res.status(400).json({ error: 'Invalid mode' });
    }
  } catch (err) {
    console.error('Handler error:', err.message);
    let userError = err.message || 'Internal server error';
    // User-friendly error messages
    if (userError.includes('429') || userError.toLowerCase().includes('quota') || userError.toLowerCase().includes('rate limit')) {
      userError = 'The AI is busy right now. Please wait 30 seconds and try again.';
    } else if (userError.includes('503') || userError.toLowerCase().includes('overloaded')) {
      userError = 'The AI service is temporarily overloaded. Please try again in a minute.';
    } else if (userError.includes('401') || userError.toLowerCase().includes('api key')) {
      userError = 'API key error. Please contact support.';
    } else if (userError.toLowerCase().includes('no image') || userError.toLowerCase().includes('no response')) {
      userError = 'Could not read the image. Try uploading a clearer photo with better lighting.';
    }
    return res.status(500).json({ error: userError });
  }
});

// ─── Correction endpoint ──────────────────────────────────
function handleCorrection(res, body) {
  const { imageHash, field, value } = body;
  if (!imageHash || !field || !value) return res.status(400).json({ error: 'Missing correction data' });

  const corrections = loadCorrections();
  if (!corrections[imageHash]) corrections[imageHash] = {};
  corrections[imageHash][field] = value;
  corrections[imageHash]._updated = new Date().toISOString();
  saveCorrections(corrections);

  console.log(`[Correction] Saved ${field}=${value} for hash ${imageHash}`);
  return res.status(200).json({ ok: true });
}

// ─── ANALYZE ──────────────────────────────────────────────
async function handleAnalyze(res, apiKey, imageParts, part, rawBase64, pdfPages) {

  // Check correction cache
  const hashSrc = pdfPages?.[0] || rawBase64 || '';
  const imgHash = hashImage(hashSrc);
  const corrections = loadCorrections();
  const cached = corrections[imgHash];

  // Preprocess images (only JPEG parts, not PDF parts)
  let processedParts = [];
  const isPdf = imageParts[0]?.inlineData?.mimeType === 'application/pdf';

  for (const p of imageParts.slice(0, 5)) {
    if (p.inlineData?.mimeType === 'application/pdf') {
      // Keep PDF as-is
      processedParts.push(p);
    } else if (p.inlineData?.mimeType === 'image/jpeg') {
      try {
        const enhanced = await preprocessForGemini(p.inlineData.data, 'full');
        processedParts.push({ inlineData: { mimeType: 'image/jpeg', data: enhanced } });
      } catch (e) {
        processedParts.push(p); // fallback to original
      }
    } else {
      processedParts.push(p);
    }
  }

  // ═══ PASS 1: Structured extraction ═══
  const pass1SystemPrompt = `You are an expert music engraver and choir director reading sheet music with extreme precision.

CRITICAL RULES:
- If any image is a title/cover page with no staves or notes, SKIP IT and look at the next image.
- Count accidentals (flats ♭ or sharps ♯) that appear between the clef symbol and the time signature. These define the key signature.
- Do NOT count accidentals that appear before individual notes (those are accidentals, not key signature).
- SATB voice identification: Soprano=top treble staff, stems pointing up. Alto=bottom treble staff, stems pointing down. Tenor=treble-8 clef (has small 8 below treble clef) or bass clef, stems up. Bass=bass clef, stems down.
- If you recognize this piece from its title, composer, or musical content, use your knowledge to verify your key signature reading.
- KEY SIGNATURE BIAS: When the piece could be either major or minor based on accidentals alone, strongly prefer major. Most choral music, folk songs, and spirituals are in major keys. Only call it minor if you are highly certain.

WATERMARKS — IGNORE COMPLETELY:
- Some sheet music has diagonal or translucent text like "For perusal purposes only", "Preview copy", "Rental material", "Not for performance", or similar watermarks.
- These watermarks are NOT part of the music. Do NOT let them affect your reading of key signatures, time signatures, note names, dynamics, or any other musical data.
- Look THROUGH the watermark text to read the actual notes and accidentals beneath it.

STARTING PITCH — TREBLE CLEF (memorize these exactly):
Lines, bottom to top: E4 (1st/bottom line), G4 (2nd line), B4 (3rd/middle line), D5 (4th line), F5 (5th/top line). Mnemonic: Every Good Boy Does Fine.
Spaces, bottom to top: F4 (1st space), A4 (2nd space), C5 (3rd space), E5 (4th/top space). Mnemonic: FACE.
- A note sitting ON a line has the line passing through the center of the note head.
- A note sitting IN a space has the note head between two lines (not touching either).
- Do NOT confuse a note on the 3rd line (B4) with a note in the 3rd space (C5) — they look close but are different notes.
- For treble-8 clef (tenor clef with small 8 below): ALL pitches are one octave lower than treble clef (E3, G3, B3, D4, F4 for lines; F3, A3, C4, E4 for spaces).

STARTING PITCH — BASS CLEF:
Lines, bottom to top: G2 (1st line), B2 (2nd line), D3 (3rd line), F3 (4th line), A3 (5th line). Mnemonic: Good Boys Do Fine Always.
Spaces, bottom to top: A2, C3, E3, G3. Mnemonic: All Cows Eat Grass.

STARTING PITCH PROCEDURE:
1. Find the vocal staff (the one with lyrics text underneath). Skip piano intro measures.
2. Find the VERY FIRST note that has a lyric syllable directly below it.
3. Count carefully: is this note ON a line or IN a space?
4. Which line or space number is it (counting from the bottom)?
5. Look up the correct pitch using the reference above. Double-check by looking at adjacent notes.`;

  const pass1UserText = `Extract all musical data for the ${part} part from this sheet music. Skip title/cover pages. Be precise about accidental counting.`;

  const pass1Raw = await callGemini(apiKey, pass1SystemPrompt, [
    { text: pass1UserText },
    ...processedParts,
  ], {
    temperature: 0,
    maxOutputTokens: 4096,
    responseSchema: ANALYZE_SCHEMA,
    thinkingBudget: 10000,
  });

  let raw;
  try {
    raw = JSON.parse(pass1Raw);
  } catch (e) {
    console.error('Schema parse failed:', e.message, pass1Raw.substring(0, 300));
    return res.status(200).json({ text: pass1Raw });
  }

  // Code-calculated key from flat/sharp count (more reliable than Gemini's key name)
  const keyResult = resolveKeyFromCounts(
    Number(raw.flat_count) || 0,
    Number(raw.sharp_count) || 0,
    raw.key_signature
  );

  // Apply cached corrections if available
  const finalKey = cached?.keySignature || keyResult.key;
  const finalPitch = cached?.startingPitch || raw.starting_pitch || 'Not determined';

  // FIX BUG5: Pre-calculate solfege from first_notes for instant preview
  const tonic = finalKey.split(' ')[0]; // e.g., "Bb" from "Bb major (2 flats)"
  const firstNotesSolfege = (raw.first_notes || []).map(n =>
    noteToSolfege(n.replace(/\d+$/, ''), tonic)
  );

  // ═══ PASS 2: Human analysis with Google Search grounding ═══
  const pass2SystemPrompt = `You are a patient, encouraging choir director writing a practice guide for a ${part} singer.
Write in a warm, supportive tone. Reference specific measures when giving tips. Be practical and specific.
If you can identify the piece, use Google Search to verify the key and get accurate composer biography and piece history.`;

  const pass2UserText = `Write a complete analysis for this ${part} singer.

Verified musical data:
- Key: ${finalKey}${!keyResult.confident && keyResult.geminiSaid !== keyResult.codeSaid ? ` (Note: visual count suggests ${keyResult.codeSaid}, AI read ${keyResult.geminiSaid} — please verify)` : ''}
- Time Signature: ${raw.time_signature}
- Tempo: ${raw.tempo}
- Starting Pitch: ${finalPitch}
- Composer: ${raw.composer_name || 'unknown'}
- Title: ${raw.piece_title || 'unknown'}
- Language: ${raw.lyrics_language || 'English'}
- First line of lyrics: "${raw.first_lyrics || 'not extracted'}"

Write a JSON response with this exact structure:
{
  "overview": "2-3 paragraphs about the piece and what the ${part} singer needs to know",
  "practiceTips": ["5-8 specific, actionable tips referencing measures where possible"],
  "composerBio": "2-3 sentences about the composer, or null if unknown",
  "pieceInfo": "historical context and performance notes, or null if unknown",
  "pronunciation": {
    "language": "${raw.lyrics_language || 'English'}",
    "needsGuide": ${(raw.lyrics_language || 'English').toLowerCase() !== 'english'},
    "words": []
  }
}

For pronunciation.words: include EVERY unique word from the visible lyrics with IPA transcription and English approximation.
Format each word as: {"word": "Ave", "ipa": "/ˈaː.ve/", "approx": "AH-veh"}
For English lyrics with no unusual words: set needsGuide to false and words to [].
Output ONLY valid JSON. No markdown code blocks.`;

  const pass2Raw = await callGemini(apiKey, pass2SystemPrompt, [
    { text: pass2UserText },
    ...processedParts.slice(0, 3),
  ], {
    temperature: 0.7,
    maxOutputTokens: 8192,
    thinkingBudget: 4000,
    tools: [{ googleSearch: {} }],
  });

  let analysis;
  try {
    const cleaned = pass2Raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    analysis = JSON.parse(cleaned);
  } catch (e) {
    console.error('Pass 2 parse failed:', e.message, pass2Raw.substring(0, 200));
    analysis = { overview: '', practiceTips: [], pronunciation: { language: 'English', needsGuide: false, words: [] } };
  }

  // Assemble final response
  const structured = {
    keySignature: finalKey,
    keyConfident: keyResult.confident,  // FIX BUG11: confidence flag for UI warning
    keyWarning: !keyResult.confident && keyResult.geminiSaid !== keyResult.codeSaid
      ? `Visual count: ${keyResult.codeSaid} | AI read: ${keyResult.geminiSaid}`
      : null,
    timeSignature: raw.time_signature || 'Not determined',
    tempo: raw.tempo === 'none' ? 'No tempo marking' : (raw.tempo || 'Not marked'),
    dynamics: raw.dynamics === 'none' ? 'None visible' : (raw.dynamics || 'None visible'),
    startingPitch: finalPitch,
    // FIX BUG6: Correct difficulty mapping
    difficulty: {
      overall: raw.difficulty_overall || 5,
      rhythm: raw.difficulty_rhythm || 4,
      range: raw.difficulty_pitch || 4,
      intervals: raw.difficulty_intervals || 4,  // FIX: was difficulty_text
      text: raw.difficulty_text || 3,            // FIX: now properly separate
    },
    // FIX BUG5: Include first notes solfege for instant preview
    firstNotesSolfege: firstNotesSolfege.length > 0 ? firstNotesSolfege : null,
    firstNotes: raw.first_notes || [],
    firstLyrics: raw.first_lyrics || null,
    overview: analysis.overview || '',
    practiceTips: Array.isArray(analysis.practiceTips) ? analysis.practiceTips : [],
    composerName: raw.composer_name && raw.composer_name !== 'unknown' ? raw.composer_name : null,
    composerBio: analysis.composerBio || null,
    pieceTitle: raw.piece_title && raw.piece_title !== 'unknown' ? raw.piece_title : null,
    pieceInfo: analysis.pieceInfo || null,
    pronunciation: analysis.pronunciation || { language: 'English', needsGuide: false, words: [] },
    _imageHash: imgHash,
  };

  return res.status(200).json({ structured, text: buildTextSummary(structured, part) });
}

function buildTextSummary(s, part) {
  return [
    `Key Signature: ${s.keySignature}${s.keyWarning ? ' ⚠️ ' + s.keyWarning : ''}`,
    `Time Signature: ${s.timeSignature}`,
    `Tempo: ${s.tempo}`,
    `Dynamics: ${s.dynamics}`,
    `Starting Pitch (${part}): ${s.startingPitch}`,
    `Difficulty Overall: ${s.difficulty.overall}/10`,
    `---`,
    `BREAKDOWN:`,
    s.overview,
    `---`,
    `PRACTICE TIPS:`,
    ...s.practiceTips.map((t, i) => `${i + 1}. ${t}`),
    s.composerName ? `COMPOSER: ${s.composerName}. ${s.composerBio || ''}` : '',
    s.pieceTitle ? `PIECE INFO: ${s.pieceTitle}. ${s.pieceInfo || ''}` : '',
  ].filter(Boolean).join('\n');
}

// ─── SOLFEGE v7: Parallel dual-extraction + self-consistency + structured JSON ─
async function handleSolfege(res, apiKey, imageParts, part) {

  // Preprocess images with binarization for maximum note clarity
  const processedParts = [];
  for (const p of imageParts.slice(0, 4)) {
    if (p.inlineData?.mimeType === 'image/jpeg') {
      try {
        const enhanced = await preprocessForGemini(p.inlineData.data, 'binarize');
        processedParts.push({ inlineData: { mimeType: 'image/jpeg', data: enhanced } });
      } catch (_) { processedParts.push(p); }
    } else {
      processedParts.push(p);
    }
  }

  // Step 1: Staff identification
  const staffRaw = await callGemini(apiKey,
    `You are reading sheet music. Which staff number (counting from the top, starting at 1) has lyrics text written below it?
For SATB: Soprano=staff 1 (top treble), Alto=staff 2 (bottom treble), Tenor=staff 3 (treble-8 or bass), Bass=staff 4 (bottom bass).
For TB choir: Tenor=staff 1 (treble-8 clef has small 8 below), Bass=staff 2 (bass clef).
Skip title/cover pages.
Output ONLY JSON: {"vocal_staff_number": N, "total_staves": M, "clef": "treble" or "bass" or "treble-8", "part_confirmed": "${part}"}`,
    [{ text: `Identify the ${part} vocal staff.` }, ...processedParts],
    { temperature: 0, maxOutputTokens: 256, thinkingBudget: 1500 }
  );

  let staffInfo = { vocal_staff_number: 1, total_staves: 1, clef: 'treble' };
  try { staffInfo = JSON.parse(staffRaw.replace(/```json?|```/gi, '').trim()); }
  catch (_) {}

  const clefRef = staffInfo.clef === 'bass'
    ? `BASS CLEF reference — Lines bottom→top: G2 B2 D3 F3 A3 (Good Boys Do Fine Always). Spaces bottom→top: A2 C3 E3 G3 (All Cows Eat Grass).`
    : staffInfo.clef === 'treble-8'
    ? `TREBLE-8 CLEF (all pitches one octave LOWER than standard treble) — Lines: E3 G3 B3 D4 F4. Spaces: F3 A3 C4 E4. Used for Tenor voices.`
    : `TREBLE CLEF reference — Lines bottom→top: E4 G4 B4 D5 F5 (Every Good Boy Does Fine). Spaces bottom→top: F4 A4 C5 E5 (FACE).
A note ON a line has the line passing through the CENTER of the note head.
A note IN a space has the note head sitting BETWEEN two lines, not touching either.
CRITICAL: The 3rd line is B4, the 3rd space is C5 — these are adjacent and look close. The 3rd line has the line through the head; the 3rd space sits between lines 3 and 4.`;

  const outputFormat = `Output ONLY a JSON array of measure objects, no other text. Each object: {"num": 1, "notes": ["C4","D4","E4"], "lyrics": "glo-ri-a"}
Rules:
- Include accidentals: Bb4, F#4, Eb5 (use letter + accidental + octave)
- Key signature accidentals apply to ALL notes of that pitch class in the piece unless cancelled
- Accidentals in the measure apply until cancelled by a natural sign
- Use [?] for any note you cannot read clearly
- Include ALL visible measures across ALL pages, even if lyrics are missing
- If a measure has tied notes, include each distinct pitch once
- Vocal range check: Soprano C4-G5, Alto G3-E5, Tenor C3-A4, Bass E2-E4 — flag anything outside this with [?]`;

  const sysBase = `You are a professional music copyist with 20 years of experience transcribing choral music.
You are looking at staff #${staffInfo.vocal_staff_number} from the top — the ${part} vocal part.
${clefRef}
${outputFormat}`;

  const sysVerify = `You are double-checking a music transcription for accuracy.
You are looking at staff #${staffInfo.vocal_staff_number} from the top — the ${part} vocal part.
${clefRef}
Read EVERY note twice before writing it. Pay special attention to:
- Notes near the middle of the staff where line/space is easiest to confuse
- Accidentals from key signature vs. natural signs
- Octave numbers (does this note look higher or lower than the one before it?)
${outputFormat}`;

  // Parse an extraction response into a measure array
  function parseExtraction(raw) {
    try {
      const cleaned = raw.replace(/```json?|```/gi, '').trim();
      const parsed = JSON.parse(cleaned);
      if (Array.isArray(parsed)) return parsed;
      if (Array.isArray(parsed.measures)) return parsed.measures;
      return [];
    } catch (_) {
      const match = raw.match(/\[[\s\S]*\]/);
      if (match) { try { return JSON.parse(match[0]); } catch (_) {} }
      return [];
    }
  }

  // Extract tonic from raw JSON text
  function extractTonic(raw) {
    const m = raw.match(/"(?:tonic|key)"\s*:\s*"([A-Ga-g][#b♯♭]?)(?:\s+(?:major|minor))?"/);
    return m ? m[1] : null;
  }

  // Step 2: Try image segmentation for tall images (multiple systems)
  // If image is tall, extract per-segment for better accuracy, then merge
  let segmentedMeasures = null;
  const firstJpeg = imageParts.find(p => p.inlineData?.mimeType === 'image/jpeg');
  if (firstJpeg) {
    const segments = await segmentImage(firstJpeg.inlineData.data, 3);
    if (segments) {
      console.log(`[Solfai] Image segmented into ${segments.length} strips for solfege`);
      try {
        const segResults = await Promise.all(segments.map((seg, idx) =>
          callGemini(apiKey, sysBase,
            [{ text: `Extract notes for ${part} from this section of the score (segment ${idx + 1} of ${segments.length}). JSON array only.` },
             { inlineData: { mimeType: 'image/jpeg', data: seg } }],
            { temperature: 0, maxOutputTokens: 4096, thinkingBudget: 3000 }
          )
        ));
        segmentedMeasures = segResults.flatMap(r => parseExtraction(r));
        // Re-number measures sequentially
        segmentedMeasures.forEach((m, i) => { m.num = i + 1; });
      } catch (e) {
        console.warn('[Solfai] Segmented extraction failed, falling back to full image:', e.message);
      }
    }
  }

  // Step 3: Run TWO full-image extractions in PARALLEL (self-consistency)
  const [ext1Raw, ext2Raw] = await Promise.all([
    callGemini(apiKey, sysBase,
      [{ text: `Extract all notes for ${part} (staff #${staffInfo.vocal_staff_number}), measure by measure. JSON array only.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 8192, thinkingBudget: 6000 }
    ),
    callGemini(apiKey, sysVerify,
      [{ text: `Verification pass — carefully re-read each note for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 8192, thinkingBudget: 6000 }
    ),
  ]);

  const measures1 = parseExtraction(ext1Raw);
  const measures2 = parseExtraction(ext2Raw);
  const tonic = extractTonic(ext1Raw) || extractTonic(ext2Raw) || 'C';

  // Step 4: Reconcile all extractions (full-image pass1, pass2, + segmented if available)
  const maxLen = Math.max(measures1.length, measures2.length, (segmentedMeasures || []).length);
  const reconciledMeasures = [];

  for (let i = 0; i < maxLen; i++) {
    const m1 = measures1[i];
    const m2 = measures2[i];
    const m3 = segmentedMeasures?.[i]; // from image segmentation

    if (!m1 && !m2 && m3) {
      // Only segmented extraction got this measure
      reconciledMeasures.push({ ...m3, confidence: 'low', disagreement: false });
      continue;
    }
    if (!m1 && m2) {
      reconciledMeasures.push({ ...m2, confidence: 'medium' });
    } else if (m1 && !m2) {
      reconciledMeasures.push({ ...m1, confidence: 'medium' });
    } else if (m1 && m2) {
      const n1 = (m1.notes || []).join(',').toLowerCase();
      const n2 = (m2.notes || []).join(',').toLowerCase();
      const n3 = m3 ? (m3.notes || []).join(',').toLowerCase() : null;
      const agree12 = n1 === n2;

      if (agree12) {
        // Both full-image passes agree — high confidence
        reconciledMeasures.push({
          num: m1.num ?? (i + 1),
          notes: m1.notes || [],
          lyrics: m1.lyrics || m2.lyrics || '',
          confidence: 'high',
          disagreement: false,
          alt_notes: null,
        });
      } else if (n3 && (n1 === n3 || n2 === n3)) {
        // Segmented extraction breaks the tie — use the one that matches
        const winner = n1 === n3 ? m1 : m2;
        reconciledMeasures.push({
          num: winner.num ?? (i + 1),
          notes: winner.notes || [],
          lyrics: winner.lyrics || m1.lyrics || m2.lyrics || '',
          confidence: 'high',
          disagreement: false,
          alt_notes: null,
        });
      } else {
        // All three disagree or no segmented data — flag for review
        reconciledMeasures.push({
          num: m1.num ?? (i + 1),
          notes: m1.notes || [],
          lyrics: m1.lyrics || m2.lyrics || '',
          confidence: 'medium',
          disagreement: true,
          alt_notes: m2.notes || [],
        });
      }
    }
  }

  // Step 5: Code-calculate solfege (never trust AI for this)
  const VALID_SOLFEGE = new Set(['Do','Di','Re','Ri','Me','Mi','Fa','Fi','Sol','Si','La','Li','Te','Ti','?']);
  for (const m of reconciledMeasures) {
    m.solfege = (m.notes || []).map(n =>
      n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), tonic)
    );
    m.valid = m.solfege.every(s => VALID_SOLFEGE.has(s));
  }

  // Build legacy text output (for backward compatibility with practice mode parser)
  let textOutput = `Key: ${tonic} (Do = ${tonic})\nStaff: ${part} (staff #${staffInfo.vocal_staff_number} of ${staffInfo.total_staves || 1})\n\n`;
  for (const m of reconciledMeasures) {
    textOutput += `m.${m.num}:\n`;
    textOutput += `  Notes:   ${(m.notes || []).join(' ')}\n`;
    textOutput += `  Solfege: ${(m.solfege || []).join(' ')}\n`;
    textOutput += `  Lyrics:  "${m.lyrics || ''}"\n\n`;
  }

  if (!reconciledMeasures.length) {
    textOutput += 'No measures could be extracted. Try uploading a clearer image.';
  }

  return res.status(200).json({
    structured: {
      key: tonic,
      tonic,
      staffNum: staffInfo.vocal_staff_number,
      totalStaves: staffInfo.total_staves || 1,
      clef: staffInfo.clef || 'treble',
      measures: reconciledMeasures,
      disagreements: reconciledMeasures.filter(m => m.disagreement).length,
    },
    text: textOutput,
  });
}

// ─── RHYTHM ───────────────────────────────────────────────
async function handleRhythm(res, apiKey, imageParts, part) {

  // Preprocess images
  const processedParts = [];
  for (const p of imageParts.slice(0, 4)) {
    if (p.inlineData?.mimeType === 'image/jpeg') {
      try {
        const enhanced = await preprocessForGemini(p.inlineData.data, 'full');
        processedParts.push({ inlineData: { mimeType: 'image/jpeg', data: enhanced } });
      } catch (_) { processedParts.push(p); }
    } else {
      processedParts.push(p);
    }
  }

  const systemPrompt = `You are a rhythm coach helping a choir singer learn their part. Be precise and practical.
Skip title/cover pages.
SATB voice identification: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.`;

  const userText = `For the ${part} voice, provide a complete rhythm guide:

1. State the time signature and what note value gets one beat.
2. For each visible measure, provide:
   m.X: Beat pattern: "1 + 2 + 3 + 4 +" | Note values: [list] | Watch out for: [any tricky rhythm]

Counting patterns to use:
- 4/4: "1 + 2 + 3 + 4 +"
- 3/4: "1 + 2 + 3 +"
- 6/8: "1-la-li 2-la-li" (or "1 + a 2 + a")
- 2/4: "1 + 2 +"
- 2/2 (cut time): "1 + 2 +"

Be thorough. Include ALL measures across all pages.`;

  const text = await callGemini(apiKey, systemPrompt,
    [{ text: userText }, ...processedParts],
    { temperature: 0, thinkingBudget: 6000 }
  );
  return res.status(200).json({ text });
}

// ─── CHAT ─────────────────────────────────────────────────
async function handleChat(res, apiKey, messages, imageParts, part) {
  const systemPrompt = `You are Solfai, a patient and encouraging choir director and music theory coach. The student sings ${part}.
Always reference the actual sheet music in your responses. Only cite content you can see in the images.
Be warm, specific, and practical. Use movable Do solfege when discussing pitches.
Never invent measures or notes that aren't in the score.
Skip title/cover pages — focus on the actual music.
SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.`;

  const contents = [];
  const chatImages = imageParts.slice(0, 3);
  let attached = false;

  if (messages?.length > 0) {
    for (const msg of messages) {
      if (msg.role === 'user') {
        const parts = [{ text: msg.parts?.[0]?.text || msg.text || '' }];
        if (!attached) { parts.push(...chatImages); attached = true; }
        contents.push({ role: 'user', parts });
      } else if (msg.role === 'model') {
        contents.push({ role: 'model', parts: [{ text: msg.parts?.[0]?.text || msg.text || '' }] });
      }
    }
  }
  if (!contents.length) {
    contents.push({ role: 'user', parts: [{ text: 'Help me with this sheet music.' }, ...chatImages] });
  }

  const url = `${BASE_URL}/${GEMINI_MODEL}:generateContent?key=${apiKey}`;

  // FIX BUG12: mediaResolution correct uppercase value in chat handler too
  const body = {
    contents,
    system_instruction: { parts: [{ text: systemPrompt }] },
    generation_config: {
      temperature: 0.7,
      max_output_tokens: 4096,
      thinking_config: { thinking_budget: 4000 },
      media_resolution: 'MEDIA_RESOLUTION_HIGH',  // FIX BUG12: was 'media_resolution_high'
    },
  };

  // Retry logic for chat too
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      let detail = `Gemini error: ${resp.status}`;
      try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}
      if ((resp.status === 429 || resp.status >= 500) && attempt < maxAttempts) {
        await new Promise(r => setTimeout(r, attempt * 2000));
        continue;
      }
      throw new Error(detail);
    }

    const data = await resp.json();
    const text = data.candidates?.[0]?.content?.parts
      ?.filter(p => p.text && !p.thought)
      .map(p => p.text).join('') || '';

    return res.status(200).json({ text: text.trim() });
  }
}

// ─── Evaluate Singing ─────────────────────────────────────
app.post('/api/evaluate-singing', upload.single('audio'), async (req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'GEMINI_API_KEY not configured' });
  if (!req.file) return res.status(400).json({ error: 'No audio file provided' });

  const filePath = req.file.path;
  try {
    const { pieceTitle, keySignature, startingPitch, solfege, selectedPart } = req.body;
    const audioBuffer = readFileSync(filePath);
    const audioBase64 = audioBuffer.toString('base64');
    const mimeType = req.file.mimetype || 'audio/webm';

    const systemPrompt = `You are a professional choir director and vocal coach evaluating a student's singing. Be encouraging, specific, and constructive. Always find something positive to say first.`;

    const userText = `Listen to this audio of a student singing. They are a ${selectedPart || 'choir'} singer.
Context: ${pieceTitle ? `Piece: "${pieceTitle}"` : 'Unknown piece'}, Key: ${keySignature || 'unknown'}, Starting pitch: ${startingPitch || 'unknown'}.
${solfege ? `Expected solfege: ${solfege}` : ''}

Evaluate their singing as a professional choir director. Return ONLY valid JSON:
{
  "overallScore": <integer 0-100>,
  "pitchAccuracy": <integer 0-100>,
  "toneQuality": <integer 0-100>,
  "breathSupport": <integer 0-100>,
  "vowelShape": <integer 0-100>,
  "rhythm": <integer 0-100>,
  "detailedFeedback": "<2-3 sentences of warm, specific, constructive feedback>",
  "actionItems": ["<specific thing to improve 1>", "<specific thing to improve 2>", "<specific thing to improve 3>"]
}

Be encouraging. Most student singers score 40-75. Only score below 30 if the recording is clearly off-pitch or unclear.`;

    const url = `${BASE_URL}/${GEMINI_MODEL}:generateContent?key=${apiKey}`;
    const body = {
      contents: [{ role: 'user', parts: [
        { text: userText },
        { inlineData: { mimeType, data: audioBase64 } }
      ]}],
      system_instruction: { parts: [{ text: systemPrompt }] },
      generation_config: {
        temperature: 0.3,
        max_output_tokens: 1024,
        response_mime_type: 'application/json',
        thinking_config: { thinking_budget: 2000 },
      },
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      let detail = `Gemini error: ${resp.status}`;
      try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}
      throw new Error(detail);
    }

    const data = await resp.json();
    const text = data.candidates?.[0]?.content?.parts
      ?.filter(p => p.text && !p.thought).map(p => p.text).join('') || '{}';

    let result;
    try { result = JSON.parse(text.replace(/```json?|```/gi, '').trim()); }
    catch (_) { result = { overallScore: 60, pitchAccuracy: 60, toneQuality: 60, breathSupport: 60, vowelShape: 60, rhythm: 60, detailedFeedback: 'Great effort! Keep practicing.', actionItems: ['Focus on pitch accuracy', 'Work on breath support', 'Practice regularly'] }; }

    return res.status(200).json(result);
  } catch (err) {
    console.error('Evaluate-singing error:', err.message);
    return res.status(500).json({ error: err.message });
  } finally {
    try { unlinkSync(filePath); } catch (_) {}
  }
});

// ─── MusicXML Parser ─────────────────────────────────────
function parseMusicXML(xmlString, targetPart) {
  // Extract part list
  const partList = [];
  const partListMatch = xmlString.match(/<part-list>([\s\S]*?)<\/part-list>/);
  if (partListMatch) {
    const partRegex = /<score-part\s+id="([^"]+)"[\s\S]*?<part-name>([^<]*)<\/part-name>/g;
    let pm;
    while ((pm = partRegex.exec(partListMatch[1])) !== null) {
      partList.push({ id: pm[1], name: pm[2].trim() });
    }
  }

  // Find the best matching part for the target voice
  const target = (targetPart || 'Soprano').toLowerCase();
  let partId = partList[0]?.id || 'P1';
  for (const p of partList) {
    if (p.name.toLowerCase().includes(target)) { partId = p.id; break; }
  }

  // Extract key signature (fifths value)
  let fifths = 0;
  const fifthsMatch = xmlString.match(/<fifths>(-?\d+)<\/fifths>/);
  if (fifthsMatch) fifths = parseInt(fifthsMatch[1], 10);

  // Resolve key from fifths
  const sharpCount = fifths > 0 ? fifths : 0;
  const flatCount = fifths < 0 ? Math.abs(fifths) : 0;
  let code = sharpCount > 0 ? `${sharpCount}s` : flatCount > 0 ? `${flatCount}b` : '0';
  const keyEntry = KEY_FROM_COUNT[code];
  const tonic = keyEntry ? keyEntry.major.split(' ')[0] : 'C';

  // Extract the matching <part> element
  const partRegex = new RegExp(`<part\\s+id="${partId}"[^>]*>([\\s\\S]*?)<\\/part>`);
  const partMatch = xmlString.match(partRegex);
  if (!partMatch) {
    return { error: `Could not find part "${targetPart}" in the MusicXML file.`, parts: partList };
  }

  const partXml = partMatch[1];
  const measures = [];

  // Parse each measure
  const measureRegex = /<measure[^>]*number="(\d+)"[^>]*>([\s\S]*?)<\/measure>/g;
  let mm;
  while ((mm = measureRegex.exec(partXml)) !== null) {
    const measureNum = parseInt(mm[1], 10);
    const measureXml = mm[2];
    const notes = [];
    const lyrics = [];

    // Parse notes within this measure
    const noteRegex = /<note>([\s\S]*?)<\/note>/g;
    let nm;
    while ((nm = noteRegex.exec(measureXml)) !== null) {
      const noteXml = nm[1];

      // Skip rests
      if (noteXml.includes('<rest')) continue;
      // Skip chord notes (secondary notes in a chord) — keep only the first
      if (noteXml.includes('<chord')) continue;

      const stepMatch = noteXml.match(/<step>([A-G])<\/step>/);
      const octaveMatch = noteXml.match(/<octave>(\d)<\/octave>/);
      const alterMatch = noteXml.match(/<alter>(-?\d+)<\/alter>/);

      if (stepMatch && octaveMatch) {
        let noteName = stepMatch[1];
        const alter = alterMatch ? parseInt(alterMatch[1], 10) : 0;
        if (alter === 1) noteName += '#';
        else if (alter === -1) noteName += 'b';
        else if (alter === 2) noteName += '##';
        else if (alter === -2) noteName += 'bb';
        noteName += octaveMatch[1];
        notes.push(noteName);
      }

      // Extract lyrics
      const lyricMatch = noteXml.match(/<lyric[\s\S]*?<text>([^<]*)<\/text>/);
      if (lyricMatch) lyrics.push(lyricMatch[1]);
    }

    if (notes.length > 0) {
      measures.push({
        num: measureNum,
        notes,
        lyrics: lyrics.join(' '),
      });
    }
  }

  return { tonic, measures, partId, partList, sharpCount, flatCount };
}

// ─── MusicXML Upload Route ───────────────────────────────
const musicxmlUpload = multer({ dest: '/tmp/solfai-uploads/', limits: { fileSize: 10 * 1024 * 1024 } });
app.post('/api/parse-musicxml', musicxmlUpload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file provided' });

  const filePath = req.file.path;
  try {
    const selectedPart = req.body.selectedPart || 'Soprano';
    let xmlString;

    // Detect .mxl (ZIP) vs .musicxml (plain XML)
    const originalName = (req.file.originalname || '').toLowerCase();
    const buf = readFileSync(filePath);

    if (originalName.endsWith('.mxl') || buf[0] === 0x50 && buf[1] === 0x4B) {
      // .mxl is a ZIP file containing .musicxml
      const zip = new AdmZip(buf);
      const entries = zip.getEntries();
      const xmlEntry = entries.find(e =>
        e.entryName.endsWith('.musicxml') || e.entryName.endsWith('.xml')
      ) || entries.find(e => !e.entryName.startsWith('META-INF') && e.entryName.endsWith('.xml'));

      if (!xmlEntry) return res.status(400).json({ error: 'No MusicXML file found inside the .mxl archive.' });
      xmlString = xmlEntry.getData().toString('utf8');
    } else {
      xmlString = buf.toString('utf8');
    }

    if (!xmlString.includes('<score-partwise') && !xmlString.includes('<score-timewise')) {
      return res.status(400).json({ error: 'This does not appear to be a valid MusicXML file.' });
    }

    const result = parseMusicXML(xmlString, selectedPart);
    if (result.error) return res.status(400).json(result);

    // Code-calculate solfege
    const VALID_SOLFEGE = new Set(['Do','Di','Re','Ri','Me','Mi','Fa','Fi','Sol','Si','La','Li','Te','Ti','?']);
    for (const m of result.measures) {
      m.solfege = (m.notes || []).map(n =>
        noteToSolfege(n.replace(/\d+$/, ''), result.tonic)
      );
      m.valid = m.solfege.every(s => VALID_SOLFEGE.has(s));
      m.confidence = 'high';
      m.disagreement = false;
    }

    console.log(`[Solfai] MusicXML parsed: ${result.measures.length} measures, part=${selectedPart}, key=${result.tonic}`);

    return res.status(200).json({
      structured: {
        key: result.tonic,
        tonic: result.tonic,
        staffNum: 1,
        totalStaves: result.partList.length,
        clef: 'treble',
        measures: result.measures,
        disagreements: 0,
        source: 'musicxml',
      },
      text: result.measures.map(m =>
        `m.${m.num}:\n  Notes:   ${m.notes.join(' ')}\n  Solfege: ${m.solfege.join(' ')}\n  Lyrics:  "${m.lyrics}"`
      ).join('\n\n'),
      parts: result.partList.map(p => p.name),
    });
  } catch (err) {
    console.error('MusicXML parse error:', err.message);
    return res.status(500).json({ error: 'Failed to parse MusicXML: ' + err.message });
  } finally {
    try { unlinkSync(filePath); } catch (_) {}
  }
});

// ─── Manual Note Entry Route ─────────────────────────────
app.post('/api/manual-solfege', (req, res) => {
  const { notes, key } = req.body;
  if (!notes) return res.status(400).json({ error: 'No notes provided' });

  const tonic = (key || 'C').replace(/\s*(major|minor)/i, '').trim();

  // Parse input: "C4 D4 E4 | F4 G4 A4 | B4 C5"
  // Pipe separates measures, space separates notes within a measure
  const rawMeasures = notes.split('|').map(s => s.trim()).filter(Boolean);
  const measures = rawMeasures.map((m, i) => {
    const noteList = m.split(/[\s,]+/).filter(Boolean);
    return { num: i + 1, notes: noteList, lyrics: '' };
  });

  // Code-calculate solfege
  const VALID_SOLFEGE = new Set(['Do','Di','Re','Ri','Me','Mi','Fa','Fi','Sol','Si','La','Li','Te','Ti','?']);
  for (const m of measures) {
    m.solfege = (m.notes || []).map(n =>
      noteToSolfege(n.replace(/\d+$/, ''), tonic)
    );
    m.valid = m.solfege.every(s => VALID_SOLFEGE.has(s));
    m.confidence = 'high';
    m.disagreement = false;
  }

  console.log(`[Solfai] Manual entry: ${measures.length} measures, key=${tonic}`);

  return res.status(200).json({
    structured: {
      key: tonic,
      tonic,
      staffNum: 1,
      totalStaves: 1,
      clef: 'treble',
      measures,
      disagreements: 0,
      source: 'manual',
    },
    text: measures.map(m =>
      `m.${m.num}:\n  Notes:   ${m.notes.join(' ')}\n  Solfege: ${m.solfege.join(' ')}`
    ).join('\n\n'),
  });
});

// ─── Start ────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Solfai v8 running on port ${PORT}`));
