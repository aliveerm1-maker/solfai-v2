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
import { readFileSync, writeFileSync, existsSync } from 'fs';
import sharp from 'sharp';

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
      enum: ["4/4","3/4","2/4","6/8","9/8","12/8","2/2","3/8","3/2","6/4","5/4","7/8"]
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

  const isMinor = (geminiKey || '').toLowerCase().includes('minor');
  const keyName = isMinor ? entry.minor : entry.major;
  const accLabel = sharpCount > 0 ? `${sharpCount} sharp${sharpCount > 1 ? 's' : ''}` :
                   flatCount > 0 ? `${flatCount} flat${flatCount > 1 ? 's' : ''}` : 'no sharps or flats';

  // FIX BUG11: Check if code-calculated key agrees with Gemini's key
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

    console.log(`[Solfai v5] mode=${mode}, part=${part}, parts=${imageParts.length}, mime=${imageMime || 'jpeg'}`);

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
    return res.status(500).json({ error: err.message || 'Internal server error' });
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
- For the starting pitch: find the VOCAL staff (the one with lyrics text underneath). Skip any piano introduction measures. Find the very first note that has a lyric syllable directly below it.
- SATB voice identification: Soprano=top treble staff, stems pointing up. Alto=bottom treble staff, stems pointing down. Tenor=treble-8 clef (has small 8 below treble clef) or bass clef, stems up. Bass=bass clef, stems down.
- If you recognize this piece from its title, composer, or musical content, use your knowledge to verify your key signature reading.`;

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

// ─── SOLFEGE (decomposed into 3 calls + code solfege) ─────
// FIX BUG7: Staff ID now sends all pages
// FIX BUG8: Images now preprocessed
async function handleSolfege(res, apiKey, imageParts, part) {

  // FIX BUG8: Preprocess images for solfege too
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

  // Step 1: Staff identification — FIX BUG7: send all pages, not just page 1
  const staffRaw = await callGemini(apiKey,
    `You are reading sheet music. Count the staves on this page. Which staff number (counting from the top, starting at 1) has lyrics text written below it?
For SATB: Soprano=staff 1 (top treble), Alto=staff 2 (bottom treble), Tenor=staff 3 (treble-8 or bass), Bass=staff 4 (bottom bass).
For TB choir: Tenor=staff 1 (treble-8 clef has small 8 below), Bass=staff 2 (bass clef).
Skip title/cover pages with no staves.
Output ONLY JSON: {"vocal_staff_number": N, "total_staves": M, "clef": "treble/bass/treble-8", "part_confirmed": "${part}"}`,
    [{ text: `Identify the ${part} vocal staff number.` }, ...processedParts],
    { temperature: 0, maxOutputTokens: 256, thinkingBudget: 2000 }
  );

  let staffInfo;
  try { staffInfo = JSON.parse(staffRaw.replace(/```json?|```/gi, '').trim()); }
  catch (_) { staffInfo = { vocal_staff_number: 1 }; }

  // Step 2: Note + lyric extraction
  const noteSystemPrompt = `You are reading sheet music with extreme precision.
Focus ONLY on staff #${staffInfo.vocal_staff_number} from the top (the ${part} vocal staff with lyrics).
For each visible measure, list:
- The note letter names with octave (e.g., 'C4', 'Bb4', 'F#5')
- The exact lyric syllable printed below each note
Skip title/cover pages. Use [?] for notes you cannot read clearly.
Be thorough — include ALL measures visible across ALL pages.`;

  const noteRaw = await callGemini(apiKey, noteSystemPrompt,
    [{ text: `Extract all notes and lyrics for ${part} (staff #${staffInfo.vocal_staff_number}). Output ONLY JSON.` }, ...processedParts],
    { temperature: 0, maxOutputTokens: 12288, thinkingBudget: 8000 }
  );

  let noteData;
  try {
    noteData = JSON.parse(noteRaw.replace(/```json?|```/gi, '').trim());
  } catch (e) {
    return res.status(200).json({ text: noteRaw });
  }

  // Step 3: Code-calculated solfege
  const tonic = (noteData.tonic || noteData.key?.split(' ')[0] || 'C').replace(/\s+/g, '');
  let output = `Key: ${noteData.key || 'Unknown'} (Do = ${tonic})\n`;
  output += `Staff: ${part} (staff #${staffInfo.vocal_staff_number} of ${staffInfo.total_staves})\n\n`;

  if (noteData.measures?.length) {
    for (const m of noteData.measures) {
      const notes = m.notes || [];
      const solfege = notes.map(n =>
        n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), tonic)
      );
      const noteStr = notes.join(' ');
      const solStr = solfege.join(' ');
      output += `m.${m.num}:\n`;
      output += `  Notes:   ${noteStr}\n`;
      output += `  Solfege: ${solStr}\n`;
      output += `  Lyrics:  "${m.lyrics || ''}"\n\n`;
    }
  } else {
    output += 'No measures could be extracted. Try uploading a clearer image.';
  }

  return res.status(200).json({ text: output });
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

// ─── Start ────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Solfai v5 running on port ${PORT}`));
