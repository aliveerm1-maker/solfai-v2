// server.js — Solfai v3: Gemini 3 Flash + tonal.js hybrid pipeline
// Phase 1: Chain-of-thought scratchpad + two-pass verification
// Phase 3: tonal.js calculates key & solfege from raw data (no Gemini guessing)

import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '20mb' }));
app.use(express.static(join(__dirname, 'public')));

// ─── Config ───────────────────────────────────────────────
const GEMINI_MODEL = 'gemini-3-flash-preview';
const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

// ─── Key signature lookup (Phase 3: code, not AI) ─────────
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

function resolveKey(flatCount, sharpCount, mode) {
  let code;
  if (sharpCount > 0) code = `${sharpCount}s`;
  else if (flatCount > 0) code = `${flatCount}b`;
  else code = '0';

  const entry = KEY_FROM_COUNT[code];
  if (!entry) return { key: 'Unknown', accidentals: code };

  const isMinor = (mode || '').toLowerCase().includes('minor');
  const keyName = isMinor ? entry.minor : entry.major;
  const accLabel = sharpCount > 0 ? `${sharpCount} sharp${sharpCount > 1 ? 's' : ''}` :
                   flatCount > 0 ? `${flatCount} flat${flatCount > 1 ? 's' : ''}` : 'no sharps or flats';

  return { key: `${keyName} (${accLabel})`, accidentals: accLabel, tonic: keyName.split(' ')[0] };
}

// ─── Solfege from note names (Phase 3: code, not AI) ──────
const SOLFEGE = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti'];
const NOTE_TO_SEMITONE = { 'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11 };

function noteToSemitone(noteName) {
  if (!noteName) return null;
  const letter = noteName[0].toUpperCase();
  let semi = NOTE_TO_SEMITONE[letter];
  if (semi == null) return null;
  for (let i = 1; i < noteName.length; i++) {
    if (noteName[i] === '#' || noteName[i] === '♯') semi++;
    if (noteName[i] === 'b' || noteName[i] === '♭') semi--;
  }
  return ((semi % 12) + 12) % 12;
}

function noteToSolfege(noteName, tonicName) {
  const tonicSemi = noteToSemitone(tonicName);
  const noteSemi = noteToSemitone(noteName);
  if (tonicSemi == null || noteSemi == null) return noteName;

  const interval = ((noteSemi - tonicSemi) % 12 + 12) % 12;
  const scaleMap = { 0: 'Do', 2: 'Re', 4: 'Mi', 5: 'Fa', 7: 'Sol', 9: 'La', 11: 'Ti',
                     1: 'Di/Ra', 3: 'Ri/Me', 6: 'Fi/Se', 8: 'Si/Le', 10: 'Li/Te' };
  return scaleMap[interval] || noteName;
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

    console.log(`[Solfai] mode=${mode}, part=${part}, images=${imageParts.length}`);

    switch (mode) {
      case 'analyze': return await handleAnalyze(res, apiKey, imageParts, part);
      case 'solfege': return await handleSolfege(res, apiKey, imageParts, part);
      case 'rhythm':  return await handleRhythm(res, apiKey, imageParts, part);
      case 'chat':    return await handleChat(res, apiKey, messages, imageParts, part);
      default:        return res.status(400).json({ error: 'Invalid mode' });
    }
  } catch (err) {
    console.error('Handler error:', err.message);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
});

// ─── Image builder ────────────────────────────────────────
function buildImageParts(imageBase64, imageMime, pdfPages) {
  const parts = [];
  if (pdfPages?.length > 0) {
    for (const page of pdfPages) {
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: page } });
    }
  } else if (imageBase64) {
    parts.push({ inlineData: { mimeType: imageMime || 'image/jpeg', data: imageBase64 } });
  }
  return parts;
}

// ─── Gemini 3 caller ──────────────────────────────────────
async function callGemini(apiKey, systemPrompt, userParts, opts = {}) {
  const {
    thinkingLevel = 'high',
    maxOutputTokens = 16384,
    temperature = 0.1,
    model = GEMINI_MODEL,
  } = opts;

  const url = `${BASE_URL}/${model}:generateContent?key=${apiKey}`;

  const body = {
    contents: [{ role: 'user', parts: userParts }],
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature,
      maxOutputTokens,
      thinkingConfig: { thinkingLevel },
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

// ─── ANALYZE (Phase 1 + 3: scratchpad + code-calculated key) ──
async function handleAnalyze(res, apiKey, imageParts, part) {

  // ═══ PASS 1: Extract raw musical data (numbers, not names) ═══
  const pass1Prompt = `You are reading sheet music from images. Extract RAW MUSICAL DATA only.
If any image is a title/cover page, SKIP IT.

Think step by step in your response. Follow these steps EXACTLY:

STEP 1 — KEY SIGNATURE:
Look at the very first staff that has musical notes. Look BETWEEN the clef symbol and the time signature.
Count EACH accidental you see:
- How many flats (♭) do you see? Write the number.
- How many sharps (♯) do you see? Write the number.
DO NOT name the key. Just count.

STEP 2 — MODE:
Does the piece sound/look major or minor? Look at the first and last chords.
Write "major" or "minor".

STEP 3 — TIME SIGNATURE:
What two numbers are stacked vertically after the key signature? (e.g., "4" over "4" = 4/4)

STEP 4 — TEMPO:
Is there a tempo marking above the first measure? (e.g., Andante, Allegro, ♩=120)
If none visible, write "none".

STEP 5 — DYNAMICS:
What dynamic markings are visible? (pp, p, mp, mf, f, ff, crescendo, diminuendo)
Note which measures they appear in.

STEP 6 — STARTING PITCH (for ${part}):
a) Find the staff that has LYRICS (words) printed underneath it. That is the vocal staff.
b) If there's a piano introduction (measures with no lyrics), skip those.
c) Find the first note that has a lyric syllable under it.
d) What syllable is under that note?
e) Is the notehead ON a line or IN a space?
f) Counting from the bottom: which line or space number? (1st, 2nd, 3rd, 4th, 5th)
g) What clef is this staff in? (treble, bass, or treble-8 if it has a small 8 below the treble clef)
h) For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up OR treble-8 clef, Bass=bottom bass stems down.
i) For TB (Tenor-Bass) choir: Tenor=treble-8 clef (treble clef with 8 below) or top voice, Bass=bass clef or bottom voice.

STEP 7 — FIRST FEW NOTES (for ${part}):
List the first 8-10 note LETTER NAMES of the ${part} vocal line, with octave numbers.
Format: ["A4", "Bb4", "C5", "A4", ...]
Only include notes you can clearly see. Use [?] for unclear notes.

STEP 8 — PIECE IDENTIFICATION:
Can you identify the piece title and composer from the score? Write them if visible.

OUTPUT FORMAT — valid JSON only, no markdown:
{
  "flatCount": 0,
  "sharpCount": 0,
  "mode": "major",
  "timeSignature": "4/4",
  "tempo": "Andante",
  "dynamics": "p at m.3",
  "startingPitchData": {
    "syllableUnder": "Ly",
    "onLineOrSpace": "space",
    "lineOrSpaceNumber": 2,
    "clef": "treble or bass or treble-8",
    "staveDescription": "2nd space from bottom in treble clef"
  },
  "firstNotes": ["A4", "Bb4", "C5"],
  "pieceTitle": "Lydia",
  "composerName": "Gabriel Fauré",
  "lyricsLanguage": "French",
  "visibleLyrics": "Lydie sur tes roses joues..."
}`;

  const limitedParts = imageParts.length > 4 ? imageParts.slice(0, 4) : imageParts;

  const pass1Raw = await callGemini(apiKey, pass1Prompt, [
    { text: `Extract raw musical data from this sheet music for the ${part} part. ${limitedParts.length} page(s). Skip title pages. Output ONLY JSON.` },
    ...limitedParts,
  ], { thinkingLevel: 'high', maxOutputTokens: 8192, temperature: 0.05 });

  let raw;
  try {
    const cleaned = pass1Raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    raw = JSON.parse(cleaned);
  } catch (e) {
    console.error('Pass 1 JSON parse failed:', e.message, pass1Raw.substring(0, 300));
    return res.status(200).json({ text: pass1Raw });
  }

  // ═══ Phase 3: Calculate key in CODE (not AI) ═══
  const keyResult = resolveKey(
    Number(raw.flatCount) || 0,
    Number(raw.sharpCount) || 0,
    raw.mode || 'major'
  );

  // ═══ Phase 3: Calculate starting pitch in CODE ═══
  const pitchData = raw.startingPitchData || {};
  const startingPitch = calculatePitch(pitchData, raw.firstNotes);

  // ═══ Phase 3: Calculate solfege from note names ═══
  let solfegePreview = '';
  if (raw.firstNotes?.length && keyResult.tonic) {
    solfegePreview = raw.firstNotes
      .map(n => n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), keyResult.tonic))
      .join(' ');
  }

  // ═══ PASS 2: Get human-readable analysis using verified data ═══
  const pass2Prompt = `You are a choir coach writing an analysis for a student who sings ${part}.

The piece has been identified as:
- Key: ${keyResult.key}
- Time Signature: ${raw.timeSignature || 'unknown'}
- Tempo: ${raw.tempo || 'unknown'}
- Starting Pitch: ${startingPitch}
- Composer: ${raw.composerName || 'unknown'}
- Title: ${raw.pieceTitle || 'unknown'}
- Language: ${raw.lyricsLanguage || 'unknown'}
- First notes solfege: ${solfegePreview || 'unknown'}

Using the sheet music images AND the data above, output valid JSON:
{
  "overview": "2-3 paragraphs for a choir student about the piece — form, texture, character, how the ${part} part fits in. Be specific about measures you can see.",
  "practiceTips": ["5-8 specific tips referencing measures"],
  "composerBio": "2-3 sentences about the composer, or null",
  "pieceInfo": "historical context, genre, performance context, or null",
  "pronunciation": {
    "language": "${raw.lyricsLanguage || 'English'}",
    "needsGuide": ${(raw.lyricsLanguage || 'English').toLowerCase() !== 'english'},
    "words": [{"word": "original", "ipa": "/ipa/", "approx": "sounds-LIKE"}]
  }
}

For pronunciation: include EVERY unique word from visible lyrics with IPA + English approximation.
For English with no unusual words → needsGuide: false, words: [].
Output ONLY valid JSON.`;

  const pass2Raw = await callGemini(apiKey, pass2Prompt, [
    { text: `Write the analysis. Output ONLY JSON.` },
    ...limitedParts,
  ], { thinkingLevel: 'medium', maxOutputTokens: 8192, temperature: 0.2 });

  let analysis;
  try {
    const cleaned = pass2Raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    analysis = JSON.parse(cleaned);
  } catch (e) {
    console.error('Pass 2 JSON parse failed:', e.message);
    analysis = { overview: '', practiceTips: [], pronunciation: { language: 'English', needsGuide: false, words: [] } };
  }

  // ═══ Assemble final structured response ═══
  const structured = {
    keySignature: keyResult.key,
    timeSignature: raw.timeSignature || 'Not determined',
    tempo: raw.tempo || 'Not marked',
    dynamics: raw.dynamics || 'None visible',
    startingPitch,
    difficulty: {
      overall: Number(raw.difficulty?.overall) || 5,
      rhythm: Number(raw.difficulty?.rhythm) || 4,
      range: Number(raw.difficulty?.range) || 4,
      intervals: Number(raw.difficulty?.intervals) || 4,
    },
    overview: analysis.overview || '',
    practiceTips: Array.isArray(analysis.practiceTips) ? analysis.practiceTips : [],
    composerName: raw.composerName || null,
    composerBio: analysis.composerBio || null,
    pieceTitle: raw.pieceTitle || null,
    pieceInfo: analysis.pieceInfo || null,
    pronunciation: analysis.pronunciation || { language: 'English', needsGuide: false, words: [] },
  };

  return res.status(200).json({ structured, text: buildTextSummary(structured, part) });
}

// ─── Calculate pitch from staff position data ─────────────
function calculatePitch(data, firstNotes) {
  const { onLineOrSpace, lineOrSpaceNumber, clef, syllableUnder, staveDescription } = data || {};
  
  // Normalize clef name — handle treble-8, treble8vb, treble_8, etc.
  const clefNorm = (clef || '').toLowerCase().replace(/[\s\-_]/g, '');
  const isTreble = clefNorm.startsWith('treble');
  const isBass = clefNorm.startsWith('bass');
  const is8vb = clefNorm.includes('8') || clefNorm.includes('8vb') || clefNorm.includes('ottava');

  if (!onLineOrSpace || !lineOrSpaceNumber || (!isTreble && !isBass)) {
    // Fallback: use firstNotes[0] if available
    if (firstNotes?.length && firstNotes[0] !== '[?]') {
      return `${firstNotes[0]}${syllableUnder ? ` (syllable '${syllableUnder}')` : ''}`;
    }
    return staveDescription || 'Not determined';
  }

  const num = Number(lineOrSpaceNumber);
  let note = null;

  if (isTreble) {
    if (onLineOrSpace === 'line') {
      const lines = { 1: 'E4', 2: 'G4', 3: 'B4', 4: 'D5', 5: 'F5' };
      note = lines[num];
    } else {
      const spaces = { 1: 'F4', 2: 'A4', 3: 'C5', 4: 'E5' };
      note = spaces[num];
    }
    // Treble 8vb (tenor clef) — everything is one octave lower
    if (is8vb && note) {
      const letter = note.replace(/\d/, '');
      const oct = parseInt(note.match(/\d/)[0]) - 1;
      note = letter + oct;
    }
  } else if (isBass) {
    if (onLineOrSpace === 'line') {
      const lines = { 1: 'G2', 2: 'B2', 3: 'D3', 4: 'F3', 5: 'A3' };
      note = lines[num];
    } else {
      const spaces = { 1: 'A2', 2: 'C3', 3: 'E3', 4: 'G3' };
      note = spaces[num];
    }
  }

  if (!note) {
    // Fallback: use firstNotes[0]
    if (firstNotes?.length && firstNotes[0] !== '[?]') {
      return `${firstNotes[0]}${syllableUnder ? ` (syllable '${syllableUnder}')` : ''}`;
    }
    return staveDescription || 'Not determined';
  }

  const clefLabel = is8vb ? 'treble 8vb clef' : `${isTreble ? 'treble' : 'bass'} clef`;
  const desc = `${note} (${ordinal(num)} ${onLineOrSpace} ${clefLabel}${syllableUnder ? `, syllable '${syllableUnder}'` : ''})`;
  return desc;
}

function ordinal(n) {
  const s = ['th','st','nd','rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
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

// ─── SOLFEGE (Phase 3: note names from AI → solfege in code) ──
async function handleSolfege(res, apiKey, imageParts, part) {
  // Pass 1: Get raw note names from Gemini
  const extractPrompt = `You are reading sheet music. Extract the NOTE NAMES for the ${part} voice, measure by measure.
Skip title/cover pages. For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.

For each visible measure, list:
- The measure number
- The note letter names with octave (e.g., A4, Bb4, C5)
- The lyrics under those notes
- Use [?] for notes you cannot read clearly

Output valid JSON:
{
  "key": "F major",
  "tonic": "F",
  "measures": [
    { "num": 1, "notes": ["A4", "Bb4", "C5", "A4"], "lyrics": "Ly-die sur tes" },
    { "num": 2, "notes": ["F4", "G4", "A4"], "lyrics": "ro-ses joues" }
  ]
}`;

  const limitedParts = imageParts.length > 4 ? imageParts.slice(0, 4) : imageParts;

  const raw = await callGemini(apiKey, extractPrompt, [
    { text: `Extract note names measure by measure for the ${part} part. Skip title pages. Output ONLY JSON.` },
    ...limitedParts,
  ], { thinkingLevel: 'high', maxOutputTokens: 12288, temperature: 0.05 });

  let data;
  try {
    const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    data = JSON.parse(cleaned);
  } catch (e) {
    // Fallback: return raw text
    return res.status(200).json({ text: raw });
  }

  // Pass 2: Convert note names → solfege in code
  const tonic = data.tonic || 'C';
  let output = `Key: ${data.key || 'Unknown'} (Do = ${tonic})\n\n`;

  if (data.measures?.length) {
    for (const m of data.measures) {
      const solfegeNotes = (m.notes || []).map(n =>
        n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), tonic)
      );
      output += `m.${m.num}: ${solfegeNotes.join(' | ')} | lyrics: "${m.lyrics || ''}"\n`;
    }
  }

  return res.status(200).json({ text: output });
}

// ─── RHYTHM ───────────────────────────────────────────────
async function handleRhythm(res, apiKey, imageParts, part) {
  const prompt = `You are a rhythm coach for choir students. Be precise.
Skip title/cover pages.
SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.

For the ${part} voice, provide:
1. Time signature and what note gets one beat.
2. For each visible measure:
   m.X: Count: "1 + 2 + 3 + 4 +" | Notes: [what ${part} sings with durations] | Tips: [tricky spots]

Use standard counting: 4/4="1 + 2 + 3 + 4 +" | 3/4="1 + 2 + 3 +" | 6/8="1-la-li 2-la-li"`;

  const limitedParts = imageParts.length > 4 ? imageParts.slice(0, 4) : imageParts;
  const text = await callGemini(apiKey, prompt, [
    { text: `Rhythm guide for ${part}. Skip title pages. Only visible measures.` },
    ...limitedParts,
  ], { thinkingLevel: 'high' });
  return res.status(200).json({ text });
}

// ─── CHAT ─────────────────────────────────────────────────
async function handleChat(res, apiKey, messages, imageParts, part) {
  const systemPrompt = `You are Solfai, a patient choir director and music theory coach. Student sings ${part}.
Reference actual sheet music. Only cite visible content. Be encouraging. Use movable Do solfege when relevant. Never invent content. Skip title pages.
SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.`;

  const contents = [];
  const chatImages = imageParts.length > 3 ? imageParts.slice(0, 3) : imageParts;
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
      temperature: 0.3, maxOutputTokens: 4096,
      thinkingConfig: { thinkingLevel: 'medium' },
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
    console.error(`Chat error: ${resp.status}`, errText.substring(0, 500));
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
app.listen(PORT, () => console.log(`Solfai v3 running on port ${PORT}`));
