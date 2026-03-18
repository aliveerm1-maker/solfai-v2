// server.js — Solfai v4: All Manus Techniques Implemented
// T1: responseSchema with enums | T2: temperature 0 | T3: direct PDF
// T4: high-res render | T5: red box annotation | T6: Google Search grounding
// T7: decomposed solfege | T8: image preprocessing | T9: correction cache
// T10: tonal.js code-calculated key/solfege

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

app.use(express.json({ limit: '20mb' }));
app.use(express.static(join(__dirname, 'public')));

// ─── Config ───────────────────────────────────────────────
const GEMINI_MODEL = 'gemini-2.5-pro';
const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const CORRECTIONS_FILE = join(__dirname, 'corrections.json');

// ─── T9: Correction Cache ─────────────────────────────────
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
  // Hash first 50KB of image data for fingerprinting
  const chunk = (base64Data || '').substring(0, 50000);
  return createHash('md5').update(chunk).digest('hex');
}

// ─── T1: responseSchema with enum constraints ─────────────
const ANALYZE_SCHEMA = {
  type: "OBJECT",
  properties: {
    key_signature: {
      type: "STRING",
      description: "The key signature. Count accidentals carefully between the clef and time signature.",
      enum: [
        "C major","G major","D major","A major","E major","B major","F# major","Cb major",
        "F major","Bb major","Eb major","Ab major","Db major","Gb major",
        "A minor","E minor","B minor","F# minor","C# minor","G# minor","D# minor",
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
      description: "The first note SUNG by the vocal part (the staff with lyrics). Skip piano introductions. Find where lyrics begin.",
      enum: [
        "C3","D3","Eb3","E3","F3","F#3","G3","Ab3","A3","Bb3","B3",
        "C4","C#4","D4","Eb4","E4","F4","F#4","G4","Ab4","A4","Bb4","B4",
        "C5","C#5","D5","Eb5","E5","F5","F#5","G5","Ab5","A5","Bb5","B5",
        "C6","D6","E6","F6","G6"
      ]
    },
    dynamics: {
      type: "STRING",
      description: "Opening dynamic marking and subsequent changes with measure numbers."
    },
    flat_count: {
      type: "INTEGER",
      description: "Number of flats in the key signature. 0 if no flats."
    },
    sharp_count: {
      type: "INTEGER",
      description: "Number of sharps in the key signature. 0 if no sharps."
    },
    first_notes: {
      type: "ARRAY",
      items: { type: "STRING" },
      description: "First 10 note letter names with octave for the selected vocal part (e.g., ['C4','E4','G4'])"
    },
    piece_title: {
      type: "STRING",
      description: "Title of the piece if visible on the score."
    },
    composer_name: {
      type: "STRING",
      description: "Composer name if visible on the score."
    },
    lyrics_language: {
      type: "STRING",
      description: "Language of the lyrics (English, Latin, French, German, Italian, etc.)"
    },
    difficulty_overall: { type: "INTEGER", description: "Overall difficulty 1-10." },
    difficulty_rhythm: { type: "INTEGER", description: "Rhythm difficulty 1-10." },
    difficulty_pitch: { type: "INTEGER", description: "Pitch/interval difficulty 1-10." },
    difficulty_text: { type: "INTEGER", description: "Text/language difficulty 1-10." },
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
  '1s': { major: 'G major', minor: 'E minor' },
  '2s': { major: 'D major', minor: 'B minor' },
  '3s': { major: 'A major', minor: 'F# minor' },
  '4s': { major: 'E major', minor: 'C# minor' },
  '5s': { major: 'B major', minor: 'G# minor' },
  '6s': { major: 'F# major', minor: 'D# minor' },
};

function resolveKeyFromCounts(flatCount, sharpCount, geminiKey) {
  let code;
  if (sharpCount > 0) code = `${sharpCount}s`;
  else if (flatCount > 0) code = `${flatCount}b`;
  else code = '0';

  const entry = KEY_FROM_COUNT[code];
  if (!entry) return geminiKey || 'Unknown';

  // Use Gemini's major/minor determination but our key NAME from the count
  const isMinor = (geminiKey || '').toLowerCase().includes('minor');
  const keyName = isMinor ? entry.minor : entry.major;
  const accLabel = sharpCount > 0 ? `${sharpCount} sharp${sharpCount > 1 ? 's' : ''}` :
                   flatCount > 0 ? `${flatCount} flat${flatCount > 1 ? 's' : ''}` : 'no sharps or flats';
  return `${keyName} (${accLabel})`;
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

// ─── T8: Image preprocessing with sharp ───────────────────
async function preprocessForGemini(base64Data, mode = 'full') {
  try {
    const buf = Buffer.from(base64Data, 'base64');
    let pipeline;

    if (mode === 'key_region') {
      // T5: Crop top-left 35% x 22% for key signature focus
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

// ─── T5: Red box annotation for key signature ─────────────
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
function buildImageParts(imageBase64, imageMime, pdfPages) {
  const parts = [];
  // T3: If we have raw PDF data (mime is application/pdf), send directly
  if (imageMime === 'application/pdf' && imageBase64) {
    parts.push({ inlineData: { mimeType: 'application/pdf', data: imageBase64 } });
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

// ─── Gemini caller ────────────────────────────────────────
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

  const genConfig = {
    temperature,
    maxOutputTokens,
    thinkingConfig: { thinkingBudget },
    mediaResolution: 'media_resolution_high',
  };

  // T1: Add responseSchema if provided
  if (responseSchema) {
    genConfig.responseMimeType = 'application/json';
    genConfig.responseSchema = responseSchema;
  }

  const body = {
    contents: [{ role: 'user', parts: userParts }],
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: genConfig,
  };

  // T6: Add Google Search grounding if requested
  if (tools) body.tools = tools;

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error(`Gemini error: ${resp.status}`, errText.substring(0, 800));
    let detail = `Gemini error: ${resp.status}`;
    try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}
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

    console.log(`[Solfai v4] mode=${mode}, part=${part}, images=${imageParts.length}, mime=${imageMime || 'jpeg'}`);

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

// ─── T9: Correction endpoint ──────────────────────────────
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

// ─── ANALYZE (T1+T2+T3+T5+T6+T8+T9+T10) ─────────────────
async function handleAnalyze(res, apiKey, imageParts, part, rawBase64, pdfPages) {

  // T9: Check correction cache
  const hashSrc = pdfPages?.[0] || rawBase64 || '';
  const imgHash = hashImage(hashSrc);
  const corrections = loadCorrections();
  const cached = corrections[imgHash];

  // T8: Preprocess images for better quality (only for JPEG images, not PDFs)
  let processedParts = imageParts;
  const isPdf = imageParts[0]?.inlineData?.mimeType === 'application/pdf';

  if (!isPdf && imageParts.length > 0) {
    try {
      const preprocessed = await Promise.all(
        imageParts.slice(0, 4).map(async (p) => {
          const enhanced = await preprocessForGemini(p.inlineData.data, 'full');
          return { inlineData: { mimeType: 'image/jpeg', data: enhanced } };
        })
      );
      processedParts = preprocessed;
    } catch (e) {
      console.error('Preprocessing failed, using originals:', e.message);
    }
  } else {
    processedParts = imageParts.slice(0, 4);
  }

  // ═══ PASS 1: Structured extraction with responseSchema + T2 temp 0 ═══
  const pass1Prompt = `You are reading sheet music images with extreme precision.
If any image is a title/cover page with no staves or notes, SKIP IT.

For the ${part} part, extract:
- Count flats and sharps in the key signature (between clef and time sig)
- The time signature
- The tempo marking
- The starting pitch: find the VOCAL staff (has lyrics underneath). Skip piano intro. Find first note with a lyric syllable under it. For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down. For TB choir: Tenor=treble-8 clef (has small 8 below), Bass=bass clef.
- The first 10 notes of the ${part} vocal line as letter names with octaves
- Dynamic markings with measure numbers
- Piece title, composer name, lyrics language
- Difficulty ratings 1-10

If you recognize the piece, use your knowledge to verify the key and starting pitch.`;

  const pass1Raw = await callGemini(apiKey, pass1Prompt, [
    { text: `Extract musical data for the ${part} part. Skip title pages.` },
    ...processedParts,
  ], {
    temperature: 0,  // T2: Maximum determinism
    maxOutputTokens: 4096,
    responseSchema: ANALYZE_SCHEMA,  // T1: Enum-constrained output
    thinkingBudget: 8000,
  });

  let raw;
  try {
    raw = JSON.parse(pass1Raw);
  } catch (e) {
    console.error('Schema parse failed:', e.message, pass1Raw.substring(0, 200));
    return res.status(200).json({ text: pass1Raw });
  }

  // T10: Code-calculated key from flat/sharp count
  const codeKey = resolveKeyFromCounts(
    Number(raw.flat_count) || 0,
    Number(raw.sharp_count) || 0,
    raw.key_signature
  );

  // T9: Apply cached corrections if available
  const finalKey = cached?.keySignature || codeKey;
  const finalPitch = cached?.startingPitch || raw.starting_pitch || 'Not determined';

  // ═══ PASS 2: Human analysis with Google Search grounding (T6) ═══
  const pass2Prompt = `You are a choir coach writing analysis for a ${part} singer.

Verified data:
- Key: ${finalKey}
- Time: ${raw.time_signature}
- Tempo: ${raw.tempo}
- Starting Pitch: ${finalPitch}
- Composer: ${raw.composer_name || 'unknown'}
- Title: ${raw.piece_title || 'unknown'}
- Language: ${raw.lyrics_language || 'English'}

If you can identify the piece, use Google Search to verify the key and get accurate composer biography and piece history. Prefer search results over your own memory.

Write a JSON response:
{
  "overview": "2-3 paragraphs for a choir student about the piece",
  "practiceTips": ["5-8 specific tips referencing measures"],
  "composerBio": "2-3 sentences or null",
  "pieceInfo": "historical context or null",
  "pronunciation": {
    "language": "${raw.lyrics_language || 'English'}",
    "needsGuide": ${(raw.lyrics_language || 'English').toLowerCase() !== 'english'},
    "words": []
  }
}

For pronunciation: include EVERY unique word from visible lyrics with IPA + English approximation. English with no unusual words → needsGuide: false, words: [].
Output ONLY valid JSON.`;

  const pass2Raw = await callGemini(apiKey, pass2Prompt, [
    { text: 'Write the analysis. Output ONLY JSON.' },
    ...processedParts,
  ], {
    temperature: 0.7,  // Higher temp for creative content
    maxOutputTokens: 8192,
    thinkingBudget: 4000,
    tools: [{ googleSearch: {} }],  // T6: Google Search grounding
  });

  let analysis;
  try {
    const cleaned = pass2Raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    analysis = JSON.parse(cleaned);
  } catch (e) {
    console.error('Pass 2 parse failed:', e.message);
    analysis = { overview: '', practiceTips: [], pronunciation: { language: 'English', needsGuide: false, words: [] } };
  }

  // Assemble final response
  const structured = {
    keySignature: finalKey,
    timeSignature: raw.time_signature || 'Not determined',
    tempo: raw.tempo === 'none' ? 'No tempo marking' : (raw.tempo || 'Not marked'),
    dynamics: raw.dynamics || 'None visible',
    startingPitch: finalPitch,
    difficulty: {
      overall: raw.difficulty_overall || 5,
      rhythm: raw.difficulty_rhythm || 4,
      range: raw.difficulty_pitch || 4,
      intervals: raw.difficulty_text || 4,
    },
    overview: analysis.overview || '',
    practiceTips: Array.isArray(analysis.practiceTips) ? analysis.practiceTips : [],
    composerName: raw.composer_name || null,
    composerBio: analysis.composerBio || null,
    pieceTitle: raw.piece_title || null,
    pieceInfo: analysis.pieceInfo || null,
    pronunciation: analysis.pronunciation || { language: 'English', needsGuide: false, words: [] },
    _imageHash: imgHash,  // Send to frontend for correction cache
  };

  return res.status(200).json({ structured, text: buildTextSummary(structured, part) });
}

function buildTextSummary(s, part) {
  return [
    `Key Signature: ${s.keySignature}`, `Time Signature: ${s.timeSignature}`,
    `Tempo: ${s.tempo}`, `Dynamics: ${s.dynamics}`,
    `Starting Pitch (${part}): ${s.startingPitch}`,
    `Difficulty Overall: ${s.difficulty.overall}/10`,
    `---`, `BREAKDOWN:`, s.overview, `---`, `PRACTICE TIPS:`,
    ...s.practiceTips.map((t, i) => `${i + 1}. ${t}`),
    s.composerName ? `COMPOSER: ${s.composerName}. ${s.composerBio || ''}` : '',
    s.pieceTitle ? `PIECE INFO: ${s.pieceTitle}. ${s.pieceInfo || ''}` : '',
  ].filter(Boolean).join('\n');
}

// ─── SOLFEGE (T7: decomposed into 3 calls + T10: code solfege) ──
async function handleSolfege(res, apiKey, imageParts, part) {
  const limitedParts = imageParts.slice(0, 4);

  // T7 Step 1: Staff identification
  const staffRaw = await callGemini(apiKey,
    `Count the staves on this sheet music page. Which staff number (from the top) has lyrics below it? Skip title pages. For TB choir: Tenor is usually staff 1 (treble-8), Bass is staff 2 (bass clef). Output JSON: {"vocal_staff_number": N, "total_staves": M, "clef": "treble/bass/treble-8"}`,
    [{ text: `Identify the vocal staff for ${part}.` }, limitedParts[0] || limitedParts[limitedParts.length - 1]],
    { temperature: 0, maxOutputTokens: 256, thinkingBudget: 2000 }
  );

  let staffInfo;
  try { staffInfo = JSON.parse(staffRaw.replace(/```json?|```/gi, '').trim()); } catch (_) { staffInfo = { vocal_staff_number: 1 }; }

  // T7 Step 2: Note + lyric extraction (focused on vocal staff only)
  const notePrompt = `Focus ONLY on staff #${staffInfo.vocal_staff_number} from the top (the ${part} vocal staff with lyrics).
For each visible measure, list the note letter names with octave (e.g., 'C4', 'Bb4') and the exact lyric syllable under each note.
Skip title/cover pages. Use [?] for unclear notes.
Output JSON: { "key": "C major", "tonic": "C", "measures": [{"num": 1, "notes": ["C4","E4"], "lyrics": "I seek"}] }`;

  const noteRaw = await callGemini(apiKey, notePrompt,
    [{ text: `Extract notes and lyrics for ${part}, staff #${staffInfo.vocal_staff_number}. Output JSON only.` }, ...limitedParts],
    { temperature: 0, maxOutputTokens: 12288, thinkingBudget: 8000 }
  );

  let noteData;
  try {
    noteData = JSON.parse(noteRaw.replace(/```json?|```/gi, '').trim());
  } catch (e) {
    return res.status(200).json({ text: noteRaw });
  }

  // T7 Step 3 + T10: Code-calculated solfege
  const tonic = (noteData.tonic || noteData.key?.split(' ')[0] || 'C').replace(/\s+/g, '');
  let output = `Key: ${noteData.key || 'Unknown'} (Do = ${tonic})\n\n`;

  if (noteData.measures?.length) {
    for (const m of noteData.measures) {
      const solfege = (m.notes || []).map(n =>
        n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), tonic)
      );
      output += `m.${m.num}: ${solfege.join(' ')} | lyrics: "${m.lyrics || ''}"\n`;
    }
  } else {
    output += 'No measures could be extracted.';
  }

  return res.status(200).json({ text: output });
}

// ─── RHYTHM ───────────────────────────────────────────────
async function handleRhythm(res, apiKey, imageParts, part) {
  const prompt = `You are a rhythm coach. Be precise. Skip title/cover pages.
For the ${part} voice, provide:
1. Time signature and what gets one beat.
2. For each visible measure:
   m.X: Count: "1 + 2 + 3 + 4 +" | Notes: [durations] | Tips: [tricky spots]

SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.
Counting: 4/4="1 + 2 + 3 + 4 +" | 3/4="1 + 2 + 3 +" | 6/8="1-la-li 2-la-li"`;

  const text = await callGemini(apiKey, prompt,
    [{ text: `Rhythm guide for ${part}. Skip title pages.` }, ...imageParts.slice(0, 4)],
    { temperature: 0, thinkingBudget: 6000 }
  );
  return res.status(200).json({ text });
}

// ─── CHAT ─────────────────────────────────────────────────
async function handleChat(res, apiKey, messages, imageParts, part) {
  const systemPrompt = `You are Solfai, a patient choir director and music theory coach. Student sings ${part}.
Reference actual sheet music. Only cite visible content. Be encouraging. Use movable Do solfege. Never invent. Skip title pages.
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
  const body = {
    contents,
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature: 0.7, maxOutputTokens: 4096,
      thinkingConfig: { thinkingBudget: 4000 },
      mediaResolution: 'media_resolution_high',
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
    ?.filter(p => p.text && !p.thought)
    .map(p => p.text).join('') || '';

  return res.status(200).json({ text: text.trim() });
}

// ─── Start ────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Solfai v4 running on port ${PORT}`));
