// server.js — Solfai v10: Maximum Accuracy Overhaul
// Architecture: Code calculates, AI extracts. All Gemini params snake_case.
// Features: 5-way consensus voting, dedicated key/pitch extraction, note-by-note
//   reconciliation, enhanced music theory validation, enharmonic awareness,
//   weighted majority voting, correction cache, vocal coach, MusicXML parser.

import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createHash } from 'crypto';
import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'fs';
import sharp from 'sharp';
import multer from 'multer';
import AdmZip from 'adm-zip';
import { Note, Scale, Interval } from 'tonal';

const upload = multer({ dest: '/tmp/solfai-uploads/' });

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '25mb' }));
app.use(express.static(join(__dirname, 'public')));

// ─── Config ───────────────────────────────────────────────
const GEMINI_MODEL = 'gemini-2.5-pro';
const GEMINI_FLASH = 'gemini-2.5-flash';
const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const CORRECTIONS_FILE = join(__dirname, 'corrections.json');

// Voting weights: Pro is more accurate, counts double
const WEIGHT_PRO = 2;
const WEIGHT_FLASH = 1;

// ─── Correction Cache ─────────────────────────────────────
function loadCorrections() {
  try {
    if (existsSync(CORRECTIONS_FILE)) {
      return JSON.parse(readFileSync(CORRECTIONS_FILE, 'utf8'));
    }
  } catch (err) {
    console.error('[Solfai] Failed to load corrections:', err.message);
  }
  return {};
}

function saveCorrections(data) {
  try {
    writeFileSync(CORRECTIONS_FILE, JSON.stringify(data, null, 2));
  } catch (err) {
    console.error('[Solfai] Failed to save corrections:', err.message);
  }
}

function hashImage(base64Data) {
  const chunk = (base64Data || '').substring(0, 50000);
  return createHash('md5').update(chunk).digest('hex');
}

// ─── Response schema with enum constraints ────────────────
const ANALYZE_SCHEMA = {
  type: "OBJECT",
  properties: {
    key_signature: {
      type: "STRING",
      description: "The key signature. Count accidentals carefully between the clef and time signature. Pick the closest match.",
      enum: [
        "C major", "G major", "D major", "A major", "E major", "B major",
        "F# major", "C# major", "Cb major",
        "F major", "Bb major", "Eb major", "Ab major", "Db major", "Gb major",
        "A minor", "E minor", "B minor", "F# minor", "C# minor", "G# minor",
        "D# minor", "A# minor",
        "D minor", "G minor", "C minor", "F minor", "Bb minor", "Eb minor", "Ab minor"
      ]
    },
    time_signature: {
      type: "STRING",
      description: "The time signature at the beginning of the piece.",
      enum: ["4/4", "3/4", "2/4", "4/8", "6/8", "9/8", "12/8", "2/2", "3/8", "3/2", "6/4", "5/4", "7/8"]
    },
    tempo: {
      type: "STRING",
      description: "The tempo marking written on the score (e.g., 'Andante', 'Allegro q=120'). Write 'none' if not visible.",
    },
    starting_pitch: {
      type: "STRING",
      description: "The first note SUNG by the vocal part (the staff with lyrics). Skip piano introductions. Find where lyrics begin. For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.",
      enum: [
        "C3", "C#3", "D3", "Eb3", "E3", "F3", "F#3", "G3", "Ab3", "A3", "Bb3", "B3",
        "C4", "C#4", "D4", "Eb4", "E4", "F4", "F#4", "G4", "Ab4", "A4", "Bb4", "B4",
        "C5", "C#5", "D5", "Eb5", "E5", "F5", "F#5", "G5", "Ab5", "A5", "Bb5", "B5",
        "C6", "D6", "E6", "F6", "G6"
      ]
    },
    dynamics: {
      type: "STRING",
      description: "Opening dynamic marking (e.g., 'mp', 'f', 'pp') and any subsequent changes. Write 'none' if not visible."
    },
    flat_count: {
      type: "INTEGER",
      description: "Number of flats in the key signature. 0 if no flats. Count each flat symbol (♭) between clef and time signature."
    },
    sharp_count: {
      type: "INTEGER",
      description: "Number of sharps in the key signature. 0 if no sharps. Count each sharp symbol (♯) between clef and time signature."
    },
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
      enum: ["English", "Latin", "German", "French", "Italian", "Spanish", "Hebrew", "Russian", "Other"]
    },
    difficulty_overall: { type: "INTEGER", description: "Overall difficulty 1-10 for a community choir singer." },
    difficulty_rhythm: { type: "INTEGER", description: "Rhythm complexity 1-10." },
    difficulty_pitch: { type: "INTEGER", description: "Pitch range difficulty 1-10." },
    difficulty_intervals: { type: "INTEGER", description: "Interval difficulty 1-10." },
    difficulty_text: { type: "INTEGER", description: "Text/language difficulty 1-10." },
  },
  required: ["key_signature", "time_signature", "starting_pitch", "dynamics", "flat_count", "sharp_count", "difficulty_overall"]
};

// Dedicated key signature extraction schema (simpler, focused)
const KEY_SIG_SCHEMA = {
  type: "OBJECT",
  properties: {
    flat_count: {
      type: "INTEGER",
      description: "Number of flat symbols (♭) between the clef and time signature. 0 if none."
    },
    sharp_count: {
      type: "INTEGER",
      description: "Number of sharp symbols (♯) between the clef and time signature. 0 if none."
    },
    confidence: {
      type: "STRING",
      enum: ["certain", "likely", "uncertain"]
    }
  },
  required: ["flat_count", "sharp_count", "confidence"]
};

// Dedicated starting pitch schema
const PITCH_SCHEMA = {
  type: "OBJECT",
  properties: {
    pitch: {
      type: "STRING",
      description: "The first sung note with octave number.",
      enum: [
        "C3", "C#3", "D3", "Eb3", "E3", "F3", "F#3", "G3", "Ab3", "A3", "Bb3", "B3",
        "C4", "C#4", "D4", "Eb4", "E4", "F4", "F#4", "G4", "Ab4", "A4", "Bb4", "B4",
        "C5", "C#5", "D5", "Eb5", "E5", "F5", "F#5", "G5", "Ab5", "A5", "Bb5", "B5",
        "C6", "D6", "E6", "F6", "G6"
      ]
    },
    line_or_space: {
      type: "STRING",
      description: "Is the note head ON a line or IN a space?",
      enum: ["on_line", "in_space"]
    },
    which_line_or_space: {
      type: "INTEGER",
      description: "Which line or space from bottom (1=lowest). Lines 1-5, Spaces 1-4."
    },
    has_accidental: {
      type: "BOOLEAN",
      description: "Does this specific note have an accidental (sharp, flat, natural) directly before it?"
    },
    measure_number: {
      type: "INTEGER",
      description: "Which measure is this note in? (1 = first measure with this vocal part)"
    }
  },
  required: ["pitch", "line_or_space", "which_line_or_space"]
};

// ─── Key from flat/sharp count (CODE, not AI) ─────────────
const KEY_FROM_COUNT = {
  '0': { major: 'C major', minor: 'A minor' },
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

  const geminiSaysMinor = (geminiKey || '').toLowerCase().includes('minor');
  const geminiMatchesExpectedMinor = geminiSaysMinor &&
    (geminiKey || '').toLowerCase().replace(/\s+/g, '') === entry.minor.toLowerCase().replace(/\s+/g, '');

  const keyName = geminiMatchesExpectedMinor ? entry.minor : entry.major;

  const accLabel = sharpCount > 0 ? `${sharpCount} sharp${sharpCount > 1 ? 's' : ''}` :
    flatCount > 0 ? `${flatCount} flat${flatCount > 1 ? 's' : ''}` : 'no sharps or flats';

  const confident = geminiKey && (
    geminiKey.toLowerCase().replace(/\s+/g, '') === keyName.toLowerCase().replace(/\s+/g, '')
  );

  return {
    key: `${keyName} (${accLabel})`,
    confident: !!confident,
    geminiSaid: geminiKey,
    codeSaid: keyName,
  };
}

// ─── Weighted Majority Voting ─────────────────────────────
// Each vote has a value and a weight. Returns the value with highest total weight.
function weightedVote(votes) {
  const tally = {};
  for (const { value, weight } of votes) {
    if (value == null || value === '' || value === undefined) continue;
    const key = String(value).trim().toLowerCase();
    if (!tally[key]) tally[key] = { value: String(value).trim(), totalWeight: 0, count: 0 };
    tally[key].totalWeight += weight;
    tally[key].count++;
  }

  let best = null;
  for (const entry of Object.values(tally)) {
    if (!best || entry.totalWeight > best.totalWeight) best = entry;
  }
  return best ? best.value : null;
}

// ─── Enharmonic Equivalence ───────────────────────────────
const ENHARMONIC_MAP = {
  'C#': 'Db', 'Db': 'C#',
  'D#': 'Eb', 'Eb': 'D#',
  'F#': 'Gb', 'Gb': 'F#',
  'G#': 'Ab', 'Ab': 'G#',
  'A#': 'Bb', 'Bb': 'A#',
  'E#': 'F', 'Fb': 'E',
  'B#': 'C', 'Cb': 'B',
};

function areEnharmonic(note1, note2) {
  if (!note1 || !note2) return false;
  const pc1 = note1.replace(/\d+$/, '');
  const pc2 = note2.replace(/\d+$/, '');
  const oct1 = note1.match(/\d+$/)?.[0];
  const oct2 = note2.match(/\d+$/)?.[0];
  if (pc1 === pc2 && oct1 === oct2) return true;

  const midi1 = Note.midi(note1);
  const midi2 = Note.midi(note2);
  if (midi1 !== null && midi2 !== null) return midi1 === midi2;

  return false;
}

// Pick the enharmonic spelling that matches the key signature
function correctEnharmonicForKey(noteName, tonic) {
  if (!noteName || !tonic) return noteName;

  const scaleInfo = Scale.get(`${tonic} major`);
  const scaleNotes = scaleInfo.notes || [];
  const scalePCs = new Set(scaleNotes.map(n => Note.get(n).pc));

  const pc = noteName.replace(/\d+$/, '');
  const oct = noteName.match(/\d+$/)?.[0] || '';

  if (scalePCs.has(pc)) return noteName;

  const enharm = ENHARMONIC_MAP[pc];
  if (enharm && scalePCs.has(enharm)) return enharm + oct;

  return noteName;
}

// ─── Solfege from note names (CODE, not AI) ───────────────
const NOTE_TO_SEMI = { 'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11 };

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
  const ts = getSemi(tonicName);
  const ns = getSemi(noteName);
  if (ts == null || ns == null) return noteName;
  const interval = ((ns - ts) % 12 + 12) % 12;
  const map = { 0: 'Do', 2: 'Re', 4: 'Mi', 5: 'Fa', 7: 'Sol', 9: 'La', 11: 'Ti', 1: 'Di', 3: 'Me', 6: 'Fi', 8: 'Si', 10: 'Te' };
  return map[interval] || noteName;
}

// ─── Note-by-Note Reconciliation ──────────────────────────
// Instead of comparing entire measure strings, compare individual notes
// across multiple extractions and pick the winner for each position.
function noteByNoteReconcile(measureSets, tonic, voicePart) {
  const maxMeasures = Math.max(...measureSets.map(s => s.length));
  const reconciled = [];
  const range = VOCAL_RANGES[voicePart] || VOCAL_RANGES['Soprano'];

  for (let mIdx = 0; mIdx < maxMeasures; mIdx++) {
    const candidates = measureSets.map(s => s[mIdx]).filter(Boolean);
    if (!candidates.length) continue;

    // Find the measure with the most notes (likely most complete)
    const maxNotes = Math.max(...candidates.map(c => (c.notes || []).length));

    const reconciledNotes = [];
    const noteConfidences = [];

    for (let nIdx = 0; nIdx < maxNotes; nIdx++) {
      const noteVotes = [];

      for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
        const notes = candidates[cIdx].notes || [];
        if (nIdx < notes.length && notes[nIdx] !== '[?]') {
          // Determine weight based on which extraction this is
          // First 3 are Pro (weight 2), last 2 are Flash (weight 1)
          const weight = cIdx < 3 ? WEIGHT_PRO : WEIGHT_FLASH;
          noteVotes.push({ value: notes[nIdx], weight });
        }
      }

      if (noteVotes.length === 0) {
        reconciledNotes.push('[?]');
        noteConfidences.push('low');
        continue;
      }

      // Group votes by enharmonic equivalence
      const groups = [];
      for (const vote of noteVotes) {
        let found = false;
        for (const group of groups) {
          if (areEnharmonic(group.canonical, vote.value)) {
            group.totalWeight += vote.weight;
            group.count++;
            found = true;
            break;
          }
        }
        if (!found) {
          groups.push({ canonical: vote.value, totalWeight: vote.weight, count: 1 });
        }
      }

      groups.sort((a, b) => b.totalWeight - a.totalWeight);
      let winner = groups[0].canonical;

      // Correct enharmonic spelling for key
      winner = correctEnharmonicForKey(winner, tonic);

      // Validate against vocal range
      const midi = Note.midi(winner);
      if (midi !== null) {
        if (midi < range.low - 3 && midi + 12 <= range.high + 3) {
          const pc = Note.get(winner).pc;
          const oct = Note.get(winner).oct;
          winner = pc + (oct + 1);
        } else if (midi > range.high + 3 && midi - 12 >= range.low - 3) {
          const pc = Note.get(winner).pc;
          const oct = Note.get(winner).oct;
          winner = pc + (oct - 1);
        }
      }

      reconciledNotes.push(winner);

      // Confidence based on agreement
      const totalVotes = noteVotes.length;
      const winnerWeight = groups[0].totalWeight;
      const totalWeight = noteVotes.reduce((s, v) => s + v.weight, 0);

      if (totalVotes >= 3 && winnerWeight >= totalWeight * 0.8) {
        noteConfidences.push('high');
      } else if (totalVotes >= 2 && winnerWeight >= totalWeight * 0.5) {
        noteConfidences.push('medium');
      } else {
        noteConfidences.push('low');
      }
    }

    // Pick best lyrics from candidates
    const lyrics = candidates.map(c => c.lyrics || '').sort((a, b) => b.length - a.length)[0] || '';

    const measureNum = candidates[0].num ?? (mIdx + 1);
    const hasDisagreement = noteConfidences.some(c => c === 'low' || c === 'medium');
    const overallConfidence = noteConfidences.every(c => c === 'high') ? 'high' :
      noteConfidences.some(c => c === 'low') ? 'low' : 'medium';

    reconciled.push({
      num: measureNum,
      notes: reconciledNotes,
      lyrics,
      confidence: overallConfidence,
      disagreement: hasDisagreement,
      noteConfidences,
    });
  }

  return reconciled;
}

// ─── Music Theory Validation (enhanced with enharmonic awareness) ──
const VOCAL_RANGES = {
  'Soprano': { low: 60, high: 79 },  // C4-G5
  'Alto': { low: 55, high: 76 },     // G3-E5
  'Tenor': { low: 48, high: 69 },    // C3-A4
  'Bass': { low: 40, high: 64 },     // E2-E4
};

function validateAndFixNotes(measures, tonic, voicePart) {
  const range = VOCAL_RANGES[voicePart] || VOCAL_RANGES['Soprano'];
  let corrections = 0;

  const scaleName = `${tonic} major`;
  const scaleInfo = Scale.get(scaleName);
  const scaleNotes = scaleInfo.notes.length > 0 ? scaleInfo.notes : [];

  const keyAccidentals = {};
  for (const sn of scaleNotes) {
    const parsed = Note.get(sn);
    if (parsed.acc) {
      keyAccidentals[parsed.letter] = parsed.acc;
    }
  }

  // Build set of valid pitch classes in the key (for diatonic checking)
  const diatonicPCs = new Set(scaleNotes.map(n => Note.get(n).pc));

  for (const m of measures) {
    if (!m.notes || !m.notes.length) continue;
    const fixed = [];

    for (let i = 0; i < m.notes.length; i++) {
      let n = m.notes[i];
      if (n === '[?]') { fixed.push(n); continue; }

      const parsed = Note.get(n);
      if (!parsed.midi) { fixed.push(n); continue; }

      // Fix 1: Enharmonic correction — use spelling that matches key
      const correctedEnharm = correctEnharmonicForKey(n, tonic);
      if (correctedEnharm !== n) {
        n = correctedEnharm;
        corrections++;
      }

      // Fix 2: Key signature accidental enforcement
      const reParsed = Note.get(n);
      if (!reParsed.acc && keyAccidentals[reParsed.letter]) {
        const corrected = reParsed.letter + keyAccidentals[reParsed.letter] + reParsed.oct;
        const corrMidi = Note.midi(corrected);
        if (corrMidi && corrMidi >= range.low - 5 && corrMidi <= range.high + 5) {
          n = corrected;
          corrections++;
        }
      }

      // Fix 3: Octave plausibility — snap to vocal range
      let midi = Note.midi(n);
      if (midi !== null) {
        if (midi < range.low - 3 && midi + 12 <= range.high + 3) {
          n = Note.get(n).pc + (Note.get(n).oct + 1);
          corrections++;
        } else if (midi > range.high + 3 && midi - 12 >= range.low - 3) {
          n = Note.get(n).pc + (Note.get(n).oct - 1);
          corrections++;
        }
      }

      // Fix 4: Interval smoothing — if jump > major 9th, try other octave
      if (i > 0 && fixed[i - 1] !== '[?]') {
        const prevMidi = Note.midi(fixed[i - 1]);
        const currMidi = Note.midi(n);
        if (prevMidi !== null && currMidi !== null) {
          const jump = Math.abs(currMidi - prevMidi);
          if (jump > 14) {
            const up = currMidi + 12;
            const down = currMidi - 12;
            const jumpUp = Math.abs(up - prevMidi);
            const jumpDown = Math.abs(down - prevMidi);
            if (jumpDown < jump && down >= range.low - 2) {
              n = Note.get(n).pc + (Note.get(n).oct - 1);
              corrections++;
            } else if (jumpUp < jump && up <= range.high + 2) {
              n = Note.get(n).pc + (Note.get(n).oct + 1);
              corrections++;
            }
          }
        }
      }

      // Fix 5: Chromatic passing tone detection
      // If a non-diatonic note appears between two diatonic notes a step apart,
      // it's likely a chromatic passing tone — leave it alone. But if it's
      // isolated and could be a misread, check if the diatonic version fits better.
      if (i > 0 && i < m.notes.length - 1) {
        const pc = Note.get(n).pc;
        if (!diatonicPCs.has(pc)) {
          const prevNote = fixed[i - 1];
          const nextNote = m.notes[i + 1];
          const prevMidi = Note.midi(prevNote);
          const nextMidi = Note.midi(nextNote);
          const currMidi = Note.midi(n);

          if (prevMidi && nextMidi && currMidi) {
            const isPassing = (
              (currMidi > prevMidi && currMidi < nextMidi) ||
              (currMidi < prevMidi && currMidi > nextMidi)
            ) && Math.abs(currMidi - prevMidi) <= 2 && Math.abs(nextMidi - currMidi) <= 2;

            // If NOT a chromatic passing tone, it might be a misread
            if (!isPassing) {
              // Check if natural version is diatonic and in range
              const natural = Note.get(n).letter + (Note.get(n).oct || '');
              const naturalMidi = Note.midi(natural);
              if (naturalMidi && diatonicPCs.has(Note.get(natural).pc) &&
                  naturalMidi >= range.low - 2 && naturalMidi <= range.high + 2) {
                // Also check with key accidental
                const withKeyAcc = keyAccidentals[Note.get(n).letter]
                  ? Note.get(n).letter + keyAccidentals[Note.get(n).letter] + Note.get(n).oct
                  : natural;
                if (diatonicPCs.has(Note.get(withKeyAcc).pc)) {
                  n = withKeyAcc;
                  corrections++;
                }
              }
            }
          }
        }
      }

      fixed.push(n);
    }

    m.notes = fixed;
    m.theoryCorrected = corrections > 0;
  }

  return corrections;
}

// ─── Starting Pitch Validation ────────────────────────────
// Cross-check the starting pitch against the key signature
function validateStartingPitch(pitch, tonic, voicePart) {
  if (!pitch || pitch === 'Not determined') return pitch;

  const range = VOCAL_RANGES[voicePart] || VOCAL_RANGES['Soprano'];
  const midi = Note.midi(pitch);

  // Check if in vocal range
  if (midi !== null) {
    if (midi < range.low - 5 || midi > range.high + 5) {
      // Way out of range — try octave correction
      if (midi < range.low && midi + 12 <= range.high + 3) {
        const pc = Note.get(pitch).pc;
        const oct = Note.get(pitch).oct;
        return pc + (oct + 1);
      } else if (midi > range.high && midi - 12 >= range.low - 3) {
        const pc = Note.get(pitch).pc;
        const oct = Note.get(pitch).oct;
        return pc + (oct - 1);
      }
    }
  }

  // Correct enharmonic spelling for key
  return correctEnharmonicForKey(pitch, tonic);
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
        .resize({ width: 1400 })
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.5 })
        .threshold(175)
        .jpeg({ quality: 97 });
    } else if (mode === 'first_system') {
      // Crop to first ~30% of height for starting pitch detection
      const meta = await sharp(buf).metadata();
      pipeline = sharp(buf)
        .extract({
          left: 0, top: 0,
          width: meta.width,
          height: Math.floor(meta.height * 0.3)
        })
        .resize({ width: 1600 })
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.0 })
        .threshold(170)
        .jpeg({ quality: 96 });
    } else if (mode === 'binarize') {
      pipeline = sharp(buf)
        .grayscale()
        .normalise()
        .sharpen({ sigma: 2.0 })
        .threshold(160)
        .jpeg({ quality: 95 });
    } else if (mode === 'high_contrast') {
      // Maximum contrast for difficult images
      pipeline = sharp(buf)
        .grayscale()
        .normalise()
        .sharpen({ sigma: 3.0 })
        .threshold(150)
        .jpeg({ quality: 97 });
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
    console.error('[Solfai] Preprocessing failed, using original:', e.message);
    return base64Data;
  }
}

// ─── Image segmentation for tall images ───────────────────
async function segmentImage(base64Data, maxSegments = 4) {
  try {
    const buf = Buffer.from(base64Data, 'base64');
    const meta = await sharp(buf).metadata();

    if (meta.height < meta.width * 0.8 || meta.height < 800) {
      return null;
    }

    const numSegments = Math.min(maxSegments, Math.ceil(meta.height / (meta.width * 0.35)));
    if (numSegments <= 1) return null;

    const overlapPx = Math.floor(meta.height * 0.03); // 3% overlap to avoid cutting through staves
    const segmentHeight = Math.ceil(meta.height / numSegments);
    const segments = [];

    for (let i = 0; i < numSegments; i++) {
      const top = Math.max(0, i * segmentHeight - (i > 0 ? overlapPx : 0));
      const height = Math.min(segmentHeight + overlapPx, meta.height - top);
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
    console.error('[Solfai] Segmentation failed:', e.message);
    return null;
  }
}

// ─── Image builder ────────────────────────────────────────
function buildImageParts(imageBase64, imageMime, pdfPages) {
  const parts = [];

  if (imageMime === 'application/pdf' && imageBase64) {
    parts.push({ inlineData: { mimeType: 'application/pdf', data: imageBase64 } });
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

// ─── Gemini caller with retry + exponential backoff ───────
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
    max_output_tokens: maxOutputTokens,
    thinking_config: { thinking_budget: thinkingBudget },
    media_resolution: 'MEDIA_RESOLUTION_HIGH',
  };

  if (responseSchema) {
    genConfig.response_mime_type = 'application/json';
    genConfig.response_schema = responseSchema;
  }

  const body = {
    contents: [{ role: 'user', parts: userParts }],
    system_instruction: { parts: [{ text: systemPrompt }] },
    generation_config: genConfig,
  };

  if (tools) body.tools = tools;

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
        try { detail = JSON.parse(errText).error?.message || detail; } catch (e) { /* ignore */ }

        if ((resp.status === 429 || resp.status >= 500) && attempt < maxAttempts) {
          const delay = attempt * 2000;
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
    if (!imageParts.length && mode !== 'correct') {
      return res.status(400).json({ error: 'No image provided' });
    }

    console.log(`[Solfai v10] mode=${mode}, part=${part}, parts=${imageParts.length}, mime=${imageMime || 'jpeg'}`);

    switch (mode) {
      case 'analyze': return await handleAnalyze(res, apiKey, imageParts, part, imageBase64, pdfPages);
      case 'solfege': return await handleSolfege(res, apiKey, imageParts, part);
      case 'rhythm': return await handleRhythm(res, apiKey, imageParts, part);
      case 'chat': return await handleChat(res, apiKey, messages, imageParts, part);
      case 'correct': return handleCorrection(res, req.body);
      default: return res.status(400).json({ error: 'Invalid mode' });
    }
  } catch (err) {
    console.error('[Solfai] Handler error:', err.message);
    let userError = err.message || 'Internal server error';
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
  if (!imageHash || !field || !value) {
    return res.status(400).json({ error: 'Missing correction data' });
  }

  const corrections = loadCorrections();
  if (!corrections[imageHash]) corrections[imageHash] = {};
  corrections[imageHash][field] = value;
  corrections[imageHash]._updated = new Date().toISOString();
  saveCorrections(corrections);

  console.log(`[Solfai] Correction saved: ${field}=${value} for hash ${imageHash}`);
  return res.status(200).json({ ok: true });
}

// ═══════════════════════════════════════════════════════════
// ANALYZE — 5-way consensus + dedicated key/pitch extraction
// ═══════════════════════════════════════════════════════════
async function handleAnalyze(res, apiKey, imageParts, part, rawBase64, pdfPages) {
  const hashSrc = pdfPages?.[0] || rawBase64 || '';
  const imgHash = hashImage(hashSrc);
  const corrections = loadCorrections();
  const cached = corrections[imgHash];

  // Preprocess images in parallel
  const processedParts = await Promise.all(
    imageParts.slice(0, 5).map(async (p) => {
      if (p.inlineData?.mimeType === 'application/pdf') return p;
      if (p.inlineData?.mimeType === 'image/jpeg') {
        try {
          const enhanced = await preprocessForGemini(p.inlineData.data, 'full');
          return { inlineData: { mimeType: 'image/jpeg', data: enhanced } };
        } catch (e) { return p; }
      }
      return p;
    })
  );

  // Prepare specialized image crops IN PARALLEL
  const firstJpeg = imageParts.find(p => p.inlineData?.mimeType === 'image/jpeg');
  let keyRegionParts = [];
  let firstSystemParts = [];

  if (firstJpeg) {
    const [keyData, firstSysData, hiContrast] = await Promise.all([
      preprocessForGemini(firstJpeg.inlineData.data, 'key_region').catch(() => null),
      preprocessForGemini(firstJpeg.inlineData.data, 'first_system').catch(() => null),
      preprocessForGemini(firstJpeg.inlineData.data, 'high_contrast').catch(() => null),
    ]);
    if (keyData) keyRegionParts = [{ inlineData: { mimeType: 'image/jpeg', data: keyData } }];
    if (firstSysData) firstSystemParts = [{ inlineData: { mimeType: 'image/jpeg', data: firstSysData } }];
    if (hiContrast) {
      // Add high-contrast version to processed parts for Flash reads
      processedParts.push({ inlineData: { mimeType: 'image/jpeg', data: hiContrast } });
    }
  }

  // ═══ PHASE 1: Dedicated key signature extraction (3 reads) ═══
  // ═══ PHASE 2: Full structured extraction (5-way consensus) ═══
  // ═══ PHASE 3: Dedicated starting pitch extraction (3 reads) ═══
  // All phases run IN PARALLEL for speed

  const keySigPrompt = `You are an expert music engraver. Count the accidentals between the CLEF SYMBOL and the TIME SIGNATURE.
These symbols define the key signature:
- Flat (♭): looks like a lowercase 'b' — count each one
- Sharp (♯): looks like a hash/pound '#' — count each one
- Do NOT count accidentals that appear before individual notes in the music
- Do NOT count naturals (♮)
- IGNORE any watermark text overlaid on the music

PROCEDURE:
1. Find the clef symbol (treble clef 𝄞 or bass clef 𝄢) at the left edge of the first staff
2. Look immediately to the RIGHT of the clef
3. Count ONLY the flat or sharp symbols BEFORE the time signature numbers
4. Report the exact count`;

  const fullExtractPrompt = `You are an expert music engraver and choir director reading sheet music with extreme precision.

CRITICAL RULES:
- If any image is a title/cover page with no staves or notes, SKIP IT and look at the next image.
- Count accidentals (flats ♭ or sharps ♯) that appear between the clef symbol and the time signature. These define the key signature.
- Do NOT count accidentals before individual notes (those are accidentals, not key signature).
- SATB voice identification: Soprano=top treble staff, stems up. Alto=bottom treble staff, stems down. Tenor=treble-8 clef or bass clef, stems up. Bass=bass clef, stems down.
- KEY SIGNATURE BIAS: When ambiguous, strongly prefer major. Most choral music is major.

WATERMARKS — IGNORE COMPLETELY:
- Diagonal or translucent text like "For perusal purposes only", "Preview copy", etc.
- Look THROUGH watermark text to read the actual notes beneath it.

STARTING PITCH — TREBLE CLEF:
Lines bottom→top: E4, G4, B4, D5, F5 (Every Good Boy Does Fine).
Spaces bottom→top: F4, A4, C5, E5 (FACE).
- A note ON a line: line passes through center of note head.
- A note IN a space: note head between two lines.
- Do NOT confuse 3rd line (B4) with 3rd space (C5).
- Middle C (C4) is on the first ledger line below treble staff.
- Treble-8 clef (tenor): ALL pitches one octave lower.

STARTING PITCH — BASS CLEF:
Lines bottom→top: G2, B2, D3, F3, A3. Spaces: A2, C3, E3, G3.
Middle C (C4) is on the first ledger line above bass staff.

STARTING PITCH PROCEDURE:
1. Find vocal staff (has lyrics underneath). Skip piano intro measures.
2. Find VERY FIRST note with a lyric syllable below it.
3. Is the note head ON a line or IN a space? Count from bottom: which line (1-5) or space (1-4)?
4. Look up the pitch from the reference above.
5. Double-check: is it within vocal range? Soprano C4-G5, Alto G3-E5, Tenor C3-A4, Bass E2-E4.`;

  const pitchPrompt = `You are a music reading specialist. Your ONLY task is to identify the FIRST SUNG NOTE.

STEP BY STEP:
1. Find the vocal staff — the staff with lyrics (text syllables) underneath the notes.
2. SKIP any piano/organ introduction measures. Look for where the TEXT starts.
3. The very first note that has a lyric syllable below it is the starting pitch.
4. Determine if this note is ON a staff line (line goes through the note head center) or IN a space (note head sits between two lines).
5. Count from the bottom: which line (1=E4, 2=G4, 3=B4, 4=D5, 5=F5) or which space (1=F4, 2=A4, 3=C5, 4=E5).
6. Check for any accidental (♯, ♭, ♮) directly before this note.
7. For treble-8 clef: subtract one octave from all pitches.
8. For bass clef: Lines 1=G2, 2=B2, 3=D3, 4=F3, 5=A3. Spaces 1=A2, 2=C3, 3=E3, 4=G3.

COMMON MISTAKES TO AVOID:
- Confusing 3rd line (B4) with 3rd space (C5) — they look similar but are different pitches
- Reading piano notes instead of vocal notes
- Missing a key signature accidental that applies to this note
- Wrong octave number

For SATB: Soprano=top treble staff stems up, Alto=bottom treble stems down, Tenor=treble-8/bass stems up, Bass=bottom bass stems down.`;

  const extractText = `Extract all musical data for the ${part} part. Skip title/cover pages. Be precise about accidental counting.`;

  // Launch ALL phases in parallel
  const allPromises = [
    // KEY SIG: 3 dedicated reads (2 Pro + 1 Flash)
    callGemini(apiKey, keySigPrompt,
      [{ text: 'Count the key signature accidentals.' }, ...(keyRegionParts.length ? keyRegionParts : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 128, responseSchema: KEY_SIG_SCHEMA, thinkingBudget: 4000 }
    ),
    callGemini(apiKey, keySigPrompt,
      [{ text: 'Second read: Count key signature accidentals again, independently.' }, ...(keyRegionParts.length ? keyRegionParts : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 128, responseSchema: KEY_SIG_SCHEMA, thinkingBudget: 4000 }
    ),
    callGemini(apiKey, keySigPrompt,
      [{ text: 'Count the flats and sharps in the key signature.' }, ...processedParts.slice(0, 1)],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 128, responseSchema: KEY_SIG_SCHEMA, thinkingBudget: 0 }
    ),

    // FULL EXTRACTION: 3 reads (2 Pro + 1 Flash) for self-consistency
    callGemini(apiKey, fullExtractPrompt,
      [{ text: extractText }, ...processedParts],
      { temperature: 0, maxOutputTokens: 4096, responseSchema: ANALYZE_SCHEMA, thinkingBudget: 12000 }
    ),
    callGemini(apiKey, fullExtractPrompt,
      [{ text: `Independent verification: ${extractText} Double-check each value.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 4096, responseSchema: ANALYZE_SCHEMA, thinkingBudget: 12000 }
    ),
    callGemini(apiKey, fullExtractPrompt,
      [{ text: `Third independent read: ${extractText}` }, ...processedParts],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 4096, responseSchema: ANALYZE_SCHEMA, thinkingBudget: 0 }
    ),

    // STARTING PITCH: 3 dedicated reads (2 Pro + 1 Flash)
    callGemini(apiKey, pitchPrompt,
      [{ text: `Find the first sung note for the ${part} part.` }, ...(firstSystemParts.length ? firstSystemParts : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 256, responseSchema: PITCH_SCHEMA, thinkingBudget: 6000 }
    ),
    callGemini(apiKey, pitchPrompt,
      [{ text: `Second read: Find the first sung note for ${part}. Be extra careful about line vs space.` }, ...(firstSystemParts.length ? firstSystemParts : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 256, responseSchema: PITCH_SCHEMA, thinkingBudget: 6000 }
    ),
    callGemini(apiKey, pitchPrompt,
      [{ text: `Find the starting pitch for ${part}.` }, ...processedParts.slice(0, 1)],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 256, responseSchema: PITCH_SCHEMA, thinkingBudget: 0 }
    ),
  ];

  const results = await Promise.all(allPromises);

  // Parse key signature votes
  const keyVotes = [];
  for (let i = 0; i < 3; i++) {
    try {
      const ks = JSON.parse(results[i]);
      const flatCount = Number(ks.flat_count) || 0;
      const sharpCount = Number(ks.sharp_count) || 0;
      const weight = i < 2 ? WEIGHT_PRO : WEIGHT_FLASH;
      keyVotes.push({ flatCount, sharpCount, weight, confidence: ks.confidence });
    } catch (e) {
      console.warn(`[Solfai] Key sig read ${i + 1} parse failed`);
    }
  }

  // Weighted vote on flat/sharp counts
  const flatCountWinner = weightedVote(keyVotes.map(v => ({ value: v.flatCount, weight: v.weight })));
  const sharpCountWinner = weightedVote(keyVotes.map(v => ({ value: v.sharpCount, weight: v.weight })));
  const votedFlats = Number(flatCountWinner) || 0;
  const votedSharps = Number(sharpCountWinner) || 0;

  console.log(`[Solfai] Key votes: ${keyVotes.map(v => `${v.flatCount}b/${v.sharpCount}s`).join(', ')} → ${votedFlats}b/${votedSharps}s`);

  // Parse full extraction results
  const fullExtractions = [];
  for (let i = 3; i < 6; i++) {
    try {
      fullExtractions.push(JSON.parse(results[i]));
    } catch (e) {
      console.warn(`[Solfai] Full extraction ${i - 2} parse failed`);
      fullExtractions.push({});
    }
  }

  // Parse starting pitch votes
  const pitchVotes = [];
  for (let i = 6; i < 9; i++) {
    try {
      const pd = JSON.parse(results[i]);
      const weight = i < 8 ? WEIGHT_PRO : WEIGHT_FLASH;
      pitchVotes.push({ value: pd.pitch, weight, lineOrSpace: pd.line_or_space, whichNum: pd.which_line_or_space });
    } catch (e) {
      console.warn(`[Solfai] Pitch read ${i - 5} parse failed`);
    }
  }

  // Also include pitch from full extractions
  for (let i = 0; i < fullExtractions.length; i++) {
    if (fullExtractions[i].starting_pitch) {
      const weight = i < 2 ? WEIGHT_PRO : WEIGHT_FLASH;
      pitchVotes.push({ value: fullExtractions[i].starting_pitch, weight: weight * 0.5 }); // lower weight since these aren't dedicated
    }
  }

  // Weighted vote on starting pitch (with enharmonic grouping)
  const pitchGroups = [];
  for (const vote of pitchVotes) {
    if (!vote.value) continue;
    let found = false;
    for (const group of pitchGroups) {
      if (areEnharmonic(group.canonical, vote.value)) {
        group.totalWeight += vote.weight;
        group.count++;
        found = true;
        break;
      }
    }
    if (!found) {
      pitchGroups.push({ canonical: vote.value, totalWeight: vote.weight, count: 1 });
    }
  }
  pitchGroups.sort((a, b) => b.totalWeight - a.totalWeight);
  const votedPitch = pitchGroups[0]?.canonical || fullExtractions[0]?.starting_pitch || 'Not determined';

  console.log(`[Solfai] Pitch votes: ${pitchVotes.map(v => v.value).join(', ')} → ${votedPitch}`);

  // Use primary Pro extraction as base, enriched by consensus
  const raw = fullExtractions[0];

  // Also get flat/sharp counts from full extractions for cross-validation
  const fullKeyVotes = fullExtractions.map((ext, i) => ({
    flatCount: Number(ext.flat_count) || 0,
    sharpCount: Number(ext.sharp_count) || 0,
    weight: i < 2 ? WEIGHT_PRO : WEIGHT_FLASH,
  }));

  // Combine dedicated key reads with full extraction key reads
  const allKeyVotes = [...keyVotes, ...fullKeyVotes];
  const finalFlats = Number(weightedVote(allKeyVotes.map(v => ({ value: v.flatCount, weight: v.weight })))) || 0;
  const finalSharps = Number(weightedVote(allKeyVotes.map(v => ({ value: v.sharpCount, weight: v.weight })))) || 0;

  console.log(`[Solfai] Combined key votes (${allKeyVotes.length} total): ${finalFlats}b/${finalSharps}s`);

  // Code-calculated key from voted flat/sharp count
  const geminiKeyVote = weightedVote(
    fullExtractions.map((ext, i) => ({
      value: ext.key_signature,
      weight: i < 2 ? WEIGHT_PRO : WEIGHT_FLASH,
    }))
  );
  const keyResult = resolveKeyFromCounts(finalFlats, finalSharps, geminiKeyVote);

  // Apply cached corrections
  const finalKey = cached?.keySignature || keyResult.key;
  const tonic = finalKey.split(' ')[0];

  // Validate starting pitch against key and range
  const validatedPitch = validateStartingPitch(votedPitch, tonic, part);
  const finalPitch = cached?.startingPitch || validatedPitch;

  // Consensus on other fields
  const votedTime = weightedVote(
    fullExtractions.map((ext, i) => ({
      value: ext.time_signature,
      weight: i < 2 ? WEIGHT_PRO : WEIGHT_FLASH,
    }))
  ) || raw.time_signature;

  const votedTempo = weightedVote(
    fullExtractions.map((ext, i) => ({
      value: ext.tempo,
      weight: i < 2 ? WEIGHT_PRO : WEIGHT_FLASH,
    }))
  ) || raw.tempo;

  // Pre-calculate solfege from first_notes (use longest first_notes array)
  const bestFirstNotes = fullExtractions
    .map(ext => ext.first_notes || [])
    .sort((a, b) => b.length - a.length)[0];
  const firstNotesSolfege = bestFirstNotes.map(n =>
    noteToSolfege(n.replace(/\d+$/, ''), tonic)
  );

  // Confidence report
  const keyAgreement = allKeyVotes.every(v => v.flatCount === finalFlats && v.sharpCount === finalSharps);
  const pitchAgreement = pitchGroups.length > 0 && pitchGroups[0].count >= 3;

  console.log(`[Solfai] Confidence: key=${keyAgreement ? 'unanimous' : 'voted'}, pitch=${pitchAgreement ? 'strong' : 'weak'}`);

  // ═══ PASS 2: Human analysis with Google Search grounding ═══
  const pass2SystemPrompt = `You are a patient, encouraging choir director writing a practice guide for a ${part} singer.
Write in a warm, supportive tone. Reference specific measures when giving tips. Be practical and specific.
If you can identify the piece, use Google Search to verify the key and get accurate composer biography and piece history.`;

  const pass2UserText = `Write a complete analysis for this ${part} singer.

Verified musical data (consensus of ${allKeyVotes.length} independent reads):
- Key: ${finalKey}${!keyResult.confident && keyResult.geminiSaid !== keyResult.codeSaid ? ` (Note: visual count suggests ${keyResult.codeSaid}, AI read ${keyResult.geminiSaid})` : ''}
- Time Signature: ${votedTime}
- Tempo: ${votedTempo}
- Starting Pitch: ${finalPitch}${pitchAgreement ? '' : ' (low confidence — verify manually)'}
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

For pronunciation.words: include EVERY unique word from visible lyrics with IPA and English approximation.
Format: {"word": "Ave", "ipa": "/ˈaː.ve/", "approx": "AH-veh"}
For English: set needsGuide to false and words to [].
Output ONLY valid JSON.`;

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
    console.error('[Solfai] Pass 2 parse failed:', e.message);
    analysis = { overview: '', practiceTips: [], pronunciation: { language: 'English', needsGuide: false, words: [] } };
  }

  const structured = {
    keySignature: finalKey,
    keyConfident: keyResult.confident || keyAgreement,
    keyWarning: !keyResult.confident && keyResult.geminiSaid !== keyResult.codeSaid
      ? `Visual count: ${keyResult.codeSaid} | AI read: ${keyResult.geminiSaid}`
      : null,
    timeSignature: votedTime || 'Not determined',
    tempo: votedTempo === 'none' ? 'No tempo marking' : (votedTempo || 'Not marked'),
    dynamics: raw.dynamics === 'none' ? 'None visible' : (raw.dynamics || 'None visible'),
    startingPitch: finalPitch,
    pitchConfident: pitchAgreement,
    difficulty: {
      overall: raw.difficulty_overall || 5,
      rhythm: raw.difficulty_rhythm || 4,
      range: raw.difficulty_pitch || 4,
      intervals: raw.difficulty_intervals || 4,
      text: raw.difficulty_text || 3,
    },
    firstNotesSolfege: firstNotesSolfege.length > 0 ? firstNotesSolfege : null,
    firstNotes: bestFirstNotes,
    firstLyrics: raw.first_lyrics || null,
    overview: analysis.overview || '',
    practiceTips: Array.isArray(analysis.practiceTips) ? analysis.practiceTips : [],
    composerName: raw.composer_name && raw.composer_name !== 'unknown' ? raw.composer_name : null,
    composerBio: analysis.composerBio || null,
    pieceTitle: raw.piece_title && raw.piece_title !== 'unknown' ? raw.piece_title : null,
    pieceInfo: analysis.pieceInfo || null,
    pronunciation: analysis.pronunciation || { language: 'English', needsGuide: false, words: [] },
    _imageHash: imgHash,
    _selfConsistency: {
      keysAgree: keyAgreement,
      pitchAgree: pitchAgreement,
      totalKeyReads: allKeyVotes.length,
      totalPitchReads: pitchVotes.length,
    },
  };

  return res.status(200).json({ structured, text: buildTextSummary(structured, part) });
}

function buildTextSummary(s, part) {
  return [
    `Key Signature: ${s.keySignature}${s.keyWarning ? ' ⚠️ ' + s.keyWarning : ''}`,
    `Time Signature: ${s.timeSignature}`,
    `Tempo: ${s.tempo}`,
    `Dynamics: ${s.dynamics}`,
    `Starting Pitch (${part}): ${s.startingPitch}${s.pitchConfident === false ? ' ⚠️ Low confidence' : ''}`,
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

// ═══════════════════════════════════════════════════════════
// SOLFEGE — 5-way extraction + note-by-note reconciliation
// ═══════════════════════════════════════════════════════════
async function handleSolfege(res, apiKey, imageParts, part) {
  const startTime = Date.now();

  // Preprocess images with binarization
  const processedParts = await Promise.all(
    imageParts.slice(0, 4).map(async (p) => {
      if (p.inlineData?.mimeType === 'image/jpeg') {
        try {
          const enhanced = await preprocessForGemini(p.inlineData.data, 'binarize');
          return { inlineData: { mimeType: 'image/jpeg', data: enhanced } };
        } catch (e) { return p; }
      }
      return p;
    })
  );

  // High contrast version for Flash reads
  const firstJpeg = imageParts.find(p => p.inlineData?.mimeType === 'image/jpeg');
  let highContrastParts = processedParts;
  if (firstJpeg) {
    try {
      const hc = await preprocessForGemini(firstJpeg.inlineData.data, 'high_contrast');
      highContrastParts = [{ inlineData: { mimeType: 'image/jpeg', data: hc } }];
    } catch (e) { /* use processed */ }
  }

  // Key region crop for key sig extraction
  let keyRegionPart = null;
  if (firstJpeg) {
    try {
      const keyData = await preprocessForGemini(firstJpeg.inlineData.data, 'key_region');
      keyRegionPart = { inlineData: { mimeType: 'image/jpeg', data: keyData } };
    } catch (e) { /* use full image */ }
  }

  // Step 1: Staff ID + key sig extraction IN PARALLEL (3 key reads)
  const [staffRaw, keySig1Raw, keySig2Raw, keySig3Raw] = await Promise.all([
    callGemini(apiKey,
      `You are reading sheet music. Which staff number (counting from top, starting at 1) has lyrics below it?
For SATB: Soprano=staff 1 (top treble), Alto=staff 2 (bottom treble), Tenor=staff 3 (treble-8 or bass), Bass=staff 4 (bottom bass).
Skip title/cover pages.
Output ONLY JSON: {"vocal_staff_number": N, "total_staves": M, "clef": "treble" or "bass" or "treble-8", "part_confirmed": "${part}"}`,
      [{ text: `Identify the ${part} vocal staff.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 256, thinkingBudget: 1500 }
    ),
    callGemini(apiKey,
      `Count the accidentals between the clef symbol and the time signature. These define the key signature.
IGNORE any watermark text overlaid on the music.
Flats look like ♭ and sharps look like ♯. Count each one carefully.
Output ONLY JSON: {"sharps": N, "flats": N}`,
      [{ text: 'Count key signature accidentals.' }, ...(keyRegionPart ? [keyRegionPart] : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 64, thinkingBudget: 3000 }
    ),
    callGemini(apiKey,
      `Count the accidentals between the clef symbol and the time signature.
Flats (♭) and sharps (♯) only. Ignore naturals and note-level accidentals.
Output ONLY JSON: {"sharps": N, "flats": N}`,
      [{ text: 'Second read: count key signature accidentals.' }, ...(keyRegionPart ? [keyRegionPart] : processedParts.slice(0, 1))],
      { temperature: 0, maxOutputTokens: 64, thinkingBudget: 3000 }
    ),
    callGemini(apiKey,
      `Count flats and sharps in the key signature (between clef and time signature).
Output ONLY JSON: {"sharps": N, "flats": N}`,
      [{ text: 'Count key signature.' }, ...processedParts.slice(0, 1)],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 64, thinkingBudget: 0 }
    ),
  ]);

  let staffInfo = { vocal_staff_number: 1, total_staves: 1, clef: 'treble' };
  try { staffInfo = JSON.parse(staffRaw.replace(/```json?|```/gi, '').trim()); }
  catch (e) { /* use defaults */ }

  // Parse and vote on key signature
  const keyReadResults = [keySig1Raw, keySig2Raw, keySig3Raw];
  const keySigVotes = [];
  for (let i = 0; i < keyReadResults.length; i++) {
    try {
      const ks = JSON.parse(keyReadResults[i].replace(/```json?|```/gi, '').trim());
      keySigVotes.push({
        flats: Number(ks.flats) || 0,
        sharps: Number(ks.sharps) || 0,
        weight: i < 2 ? WEIGHT_PRO : WEIGHT_FLASH,
      });
    } catch (e) { /* skip */ }
  }

  const votedFlats = Number(weightedVote(keySigVotes.map(v => ({ value: v.flats, weight: v.weight })))) || 0;
  const votedSharps = Number(weightedVote(keySigVotes.map(v => ({ value: v.sharps, weight: v.weight })))) || 0;

  const keyResult = resolveKeyFromCounts(votedFlats, votedSharps, null);
  const extractedKey = keyResult.codeSaid || keyResult.key;
  console.log(`[Solfai] Key sig voted: ${keySigVotes.map(v => `${v.flats}b/${v.sharps}s`).join(', ')} → ${extractedKey}`);

  const clefRef = staffInfo.clef === 'bass'
    ? `BASS CLEF — Lines bottom→top: G2 B2 D3 F3 A3. Spaces: A2 C3 E3 G3.`
    : staffInfo.clef === 'treble-8'
      ? `TREBLE-8 CLEF (all pitches one octave LOWER than treble) — Lines: E3 G3 B3 D4 F4. Spaces: F3 A3 C4 E4.`
      : `TREBLE CLEF — Lines bottom→top: E4 G4 B4 D5 F5. Spaces: F4 A4 C5 E5.
A note ON a line: line through center of head. A note IN a space: between two lines.
CRITICAL: 3rd line = B4, 3rd space = C5 — they look close but differ.
Middle C (C4) = first ledger line below staff.`;

  const scaleNotes = extractedKey ? Scale.get(extractedKey + ' major').notes.join(', ') : '';
  const keyConstraint = extractedKey
    ? `\nKEY SIGNATURE CONSTRAINT: This piece is in ${extractedKey}. Scale notes: ${scaleNotes}. Every note with a key sig accidental MUST include it unless cancelled by a natural sign (♮).`
    : '';

  const outputFormat = `Output ONLY a JSON array of measure objects. Each: {"num": 1, "notes": ["C4","D4","E4"], "lyrics": "glo-ri-a"}
Rules:
- Include accidentals: Bb4, F#4, Eb5
- Key signature accidentals apply to ALL notes of that pitch class unless cancelled by natural
- Use [?] for unreadable notes
- Include ALL visible measures across ALL pages
- Vocal range check: Soprano C4-G5, Alto G3-E5, Tenor C3-A4, Bass E2-E4
- IGNORE watermark text completely${keyConstraint}`;

  const sysBase = `You are a professional music copyist transcribing choral music with extreme precision.
Staff #${staffInfo.vocal_staff_number} from the top — ${part} vocal part.
${clefRef}

READING PROCEDURE FOR EACH NOTE:
1. Is the note head ON a line or IN a space?
2. Count from the bottom: which line (1-5) or space (1-4)?
3. Look up the pitch name from the clef reference above
4. Check for accidentals directly before the note (♯ ♭ ♮)
5. Apply key signature accidentals if no natural cancels them
6. Determine octave number from staff position

${outputFormat}`;

  const sysVerify = `You are double-checking a music transcription with extreme care.
Staff #${staffInfo.vocal_staff_number} from the top — ${part} vocal part.
${clefRef}
Read EVERY note twice. Pay special attention to:
- Notes near middle of staff where line/space is confusable (B4 vs C5)
- Key signature accidentals vs natural signs
- Octave numbers — count ledger lines carefully
- Tied notes (count only once) vs repeated notes
${outputFormat}`;

  function parseExtraction(raw) {
    try {
      const cleaned = raw.replace(/```json?|```/gi, '').trim();
      const parsed = JSON.parse(cleaned);
      if (Array.isArray(parsed)) return parsed;
      if (Array.isArray(parsed.measures)) return parsed.measures;
      return [];
    } catch (e) {
      const match = raw.match(/\[[\s\S]*\]/);
      if (match) { try { return JSON.parse(match[0]); } catch (e2) { /* ignore */ } }
      return [];
    }
  }

  // Step 2: Try image segmentation for tall images
  let segmentedMeasures = null;
  if (firstJpeg) {
    const segments = await segmentImage(firstJpeg.inlineData.data, 4);
    if (segments) {
      console.log(`[Solfai] Image segmented into ${segments.length} strips`);
      try {
        const segResults = await Promise.all(segments.map((seg, idx) =>
          callGemini(apiKey, sysBase,
            [{ text: `Extract notes for ${part} from segment ${idx + 1} of ${segments.length}. JSON array only.` },
            { inlineData: { mimeType: 'image/jpeg', data: seg } }],
            { temperature: 0, maxOutputTokens: 4096, thinkingBudget: 4000 }
          )
        ));
        segmentedMeasures = segResults.flatMap(r => parseExtraction(r));
        // Re-number measures sequentially
        segmentedMeasures.forEach((m, i) => { m.num = i + 1; });
      } catch (e) {
        console.warn('[Solfai] Segmented extraction failed:', e.message);
      }
    }
  }

  // Step 3: FIVE extractions in PARALLEL (3x Pro + 2x Flash)
  const extractionPromises = [
    // Pro extraction 1
    callGemini(apiKey, sysBase,
      [{ text: `Extract all notes for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 8192, thinkingBudget: 8000 }
    ),
    // Pro extraction 2 (verification)
    callGemini(apiKey, sysVerify,
      [{ text: `Verification pass — carefully re-read each note for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` }, ...processedParts],
      { temperature: 0, maxOutputTokens: 8192, thinkingBudget: 8000 }
    ),
    // Pro extraction 3 (with high contrast image)
    callGemini(apiKey, sysBase,
      [{ text: `Third independent read: Extract all notes for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` },
       ...(highContrastParts.length ? highContrastParts : processedParts)],
      { temperature: 0, maxOutputTokens: 8192, thinkingBudget: 8000 }
    ),
    // Flash extraction 1
    callGemini(apiKey, sysBase,
      [{ text: `Extract all notes for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` }, ...processedParts],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 4096, thinkingBudget: 0 }
    ),
    // Flash extraction 2
    callGemini(apiKey, sysVerify,
      [{ text: `Verify and extract all notes for ${part} (staff #${staffInfo.vocal_staff_number}). JSON array only.` },
       ...(highContrastParts.length ? highContrastParts : processedParts)],
      { model: GEMINI_FLASH, temperature: 0, maxOutputTokens: 4096, thinkingBudget: 0 }
    ),
  ];

  const extractionResults = await Promise.all(extractionPromises);

  const measureSets = extractionResults.map(r => parseExtraction(r));
  // Include segmented results if available
  if (segmentedMeasures?.length) measureSets.push(segmentedMeasures);

  const tonic = extractedKey?.split(' ')[0] || 'C';
  console.log(`[Solfai] Extractions: ${measureSets.map((s, i) => `${i < 3 ? 'Pro' : i < 5 ? 'Flash' : 'Seg'}=${s.length}m`).join(', ')}, tonic=${tonic}`);

  // Step 4: Note-by-note reconciliation across ALL extractions
  const reconciledMeasures = noteByNoteReconcile(measureSets, tonic, part);

  // Step 5: Music theory validation
  const theoryCorrections = validateAndFixNotes(reconciledMeasures, tonic, part);
  if (theoryCorrections > 0) {
    console.log(`[Solfai] Theory validation corrected ${theoryCorrections} notes`);
  }

  // Step 6: Code-calculate solfege
  const VALID_SOLFEGE = new Set(['Do', 'Di', 'Re', 'Ri', 'Me', 'Mi', 'Fa', 'Fi', 'Sol', 'Si', 'La', 'Li', 'Te', 'Ti', '?']);
  for (const m of reconciledMeasures) {
    m.solfege = (m.notes || []).map(n =>
      n === '[?]' ? '?' : noteToSolfege(n.replace(/\d+$/, ''), tonic)
    );
    m.valid = m.solfege.every(s => VALID_SOLFEGE.has(s));
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const highConfCount = reconciledMeasures.filter(m => m.confidence === 'high').length;
  console.log(`[Solfai] Solfege complete: ${reconciledMeasures.length} measures (${highConfCount} high-conf) in ${elapsed}s`);

  // Build text output
  let textOutput = `Key: ${tonic} (Do = ${tonic})\nStaff: ${part} (staff #${staffInfo.vocal_staff_number} of ${staffInfo.total_staves || 1})\n\n`;
  for (const m of reconciledMeasures) {
    textOutput += `m.${m.num}:\n`;
    textOutput += `  Notes:   ${(m.notes || []).join(' ')}\n`;
    textOutput += `  Solfege: ${(m.solfege || []).join(' ')}\n`;
    textOutput += `  Lyrics:  "${m.lyrics || ''}"\n`;
    if (m.confidence !== 'high') textOutput += `  ⚠ Confidence: ${m.confidence}\n`;
    textOutput += '\n';
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
      theoryCorrections,
      extractionCount: measureSets.length,
      highConfidenceCount: highConfCount,
    },
    text: textOutput,
  });
}

// ─── RHYTHM ───────────────────────────────────────────────
async function handleRhythm(res, apiKey, imageParts, part) {
  const processedParts = await Promise.all(
    imageParts.slice(0, 4).map(async (p) => {
      if (p.inlineData?.mimeType === 'image/jpeg') {
        try {
          const enhanced = await preprocessForGemini(p.inlineData.data, 'full');
          return { inlineData: { mimeType: 'image/jpeg', data: enhanced } };
        } catch (e) { return p; }
      }
      return p;
    })
  );

  const systemPrompt = `You are a rhythm coach helping a choir singer learn their part.
Skip title/cover pages. IGNORE watermark text.
SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass/treble-8 stems up, Bass=bottom bass stems down.`;

  const userText = `For the ${part} voice, provide a complete rhythm guide:

1. State the time signature and what note value gets one beat.
2. For each visible measure:
   m.X: Beat pattern: "1 + 2 + 3 + 4 +" | Note values: [list] | Watch out for: [tricky rhythm]

Counting patterns:
- 4/4: "1 + 2 + 3 + 4 +"
- 3/4: "1 + 2 + 3 +"
- 6/8: "1-la-li 2-la-li"
- 2/4: "1 + 2 +"
- 2/2 (cut time): "1 + 2 +"
- 4/8: "1-e-+ 2-e-+"

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
Always reference the actual sheet music. Only cite content you can see in the images.
Be warm, specific, and practical. Use movable Do solfege.
Never invent measures or notes not in the score.
Skip title/cover pages. IGNORE watermark text.
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
    system_instruction: { parts: [{ text: systemPrompt }] },
    generation_config: {
      temperature: 0.7,
      max_output_tokens: 4096,
      thinking_config: { thinking_budget: 4000 },
      media_resolution: 'MEDIA_RESOLUTION_HIGH',
    },
  };

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
      try { detail = JSON.parse(errText).error?.message || detail; } catch (e) { /* ignore */ }
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

    const systemPrompt = `You are a professional choir director and vocal coach evaluating a student's singing. Be encouraging, specific, and constructive. Always find something positive first.`;

    const userText = `Listen to this audio of a student singing. They are a ${selectedPart || 'choir'} singer.
Context: ${pieceTitle ? `Piece: "${pieceTitle}"` : 'Unknown piece'}, Key: ${keySignature || 'unknown'}, Starting pitch: ${startingPitch || 'unknown'}.
${solfege ? `Expected solfege: ${solfege}` : ''}

Evaluate as a professional choir director. Return ONLY valid JSON:
{
  "overallScore": <integer 0-100>,
  "pitchAccuracy": <integer 0-100>,
  "toneQuality": <integer 0-100>,
  "breathSupport": <integer 0-100>,
  "vowelShape": <integer 0-100>,
  "rhythm": <integer 0-100>,
  "diction": <integer 0-100>,
  "detailedFeedback": "<2-3 sentences of warm, specific, constructive feedback>",
  "actionItems": ["<specific improvement 1>", "<specific improvement 2>", "<specific improvement 3>"],
  "strengths": ["<positive observation 1>", "<positive observation 2>"]
}

Most student singers score 40-75. Only score below 30 if clearly off-pitch or unclear.`;

    const url = `${BASE_URL}/${GEMINI_MODEL}:generateContent?key=${apiKey}`;
    const body = {
      contents: [{ role: 'user', parts: [
        { text: userText },
        { inlineData: { mimeType, data: audioBase64 } }
      ] }],
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
      try { detail = JSON.parse(errText).error?.message || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }

    const data = await resp.json();
    const text = data.candidates?.[0]?.content?.parts
      ?.filter(p => p.text && !p.thought).map(p => p.text).join('') || '{}';

    let result;
    try {
      result = JSON.parse(text.replace(/```json?|```/gi, '').trim());
    } catch (e) {
      result = {
        overallScore: 60, pitchAccuracy: 60, toneQuality: 60,
        breathSupport: 60, vowelShape: 60, rhythm: 60, diction: 60,
        detailedFeedback: 'Great effort! Keep practicing.',
        actionItems: ['Focus on pitch accuracy', 'Work on breath support', 'Practice regularly'],
        strengths: ['Good effort', 'Consistent rhythm']
      };
    }

    return res.status(200).json(result);
  } catch (err) {
    console.error('[Solfai] Evaluate-singing error:', err.message);
    return res.status(500).json({ error: err.message });
  } finally {
    try { unlinkSync(filePath); } catch (e) { /* ignore */ }
  }
});

// ─── MusicXML Parser ──────────────────────────────────────
function parseMusicXML(xmlString, targetPart) {
  const partList = [];
  const partListMatch = xmlString.match(/<part-list>([\s\S]*?)<\/part-list>/);
  if (partListMatch) {
    const partRegex = /<score-part\s+id="([^"]+)"[\s\S]*?<part-name>([^<]*)<\/part-name>/g;
    let pm;
    while ((pm = partRegex.exec(partListMatch[1])) !== null) {
      partList.push({ id: pm[1], name: pm[2].trim() });
    }
  }

  const target = (targetPart || 'Soprano').toLowerCase();
  let partId = partList[0]?.id || 'P1';
  for (const p of partList) {
    if (p.name.toLowerCase().includes(target)) { partId = p.id; break; }
  }

  let fifths = 0;
  const fifthsMatch = xmlString.match(/<fifths>(-?\d+)<\/fifths>/);
  if (fifthsMatch) fifths = parseInt(fifthsMatch[1], 10);

  const sharpCount = fifths > 0 ? fifths : 0;
  const flatCount = fifths < 0 ? Math.abs(fifths) : 0;
  const code = sharpCount > 0 ? `${sharpCount}s` : flatCount > 0 ? `${flatCount}b` : '0';
  const keyEntry = KEY_FROM_COUNT[code];
  const tonic = keyEntry ? keyEntry.major.split(' ')[0] : 'C';

  const partRegex = new RegExp(`<part\\s+id="${partId}"[^>]*>([\\s\\S]*?)<\\/part>`);
  const partMatch = xmlString.match(partRegex);
  if (!partMatch) {
    return { error: `Could not find part "${targetPart}" in the MusicXML file.`, parts: partList };
  }

  const partXml = partMatch[1];
  const measures = [];

  const measureRegex = /<measure[^>]*number="(\d+)"[^>]*>([\s\S]*?)<\/measure>/g;
  let mm;
  while ((mm = measureRegex.exec(partXml)) !== null) {
    const measureNum = parseInt(mm[1], 10);
    const measureXml = mm[2];
    const notes = [];
    const lyrics = [];

    const noteRegex = /<note>([\s\S]*?)<\/note>/g;
    let nm;
    while ((nm = noteRegex.exec(measureXml)) !== null) {
      const noteXml = nm[1];
      if (noteXml.includes('<rest')) continue;
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

      const lyricMatch = noteXml.match(/<lyric[\s\S]*?<text>([^<]*)<\/text>/);
      if (lyricMatch) lyrics.push(lyricMatch[1]);
    }

    if (notes.length > 0) {
      measures.push({ num: measureNum, notes, lyrics: lyrics.join(' ') });
    }
  }

  return { tonic, measures, partId, partList, sharpCount, flatCount };
}

// ─── MusicXML Upload Route ────────────────────────────────
const musicxmlUpload = multer({ dest: '/tmp/solfai-uploads/', limits: { fileSize: 10 * 1024 * 1024 } });
app.post('/api/parse-musicxml', musicxmlUpload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file provided' });

  const filePath = req.file.path;
  try {
    const selectedPart = req.body.selectedPart || 'Soprano';
    let xmlString;

    const originalName = (req.file.originalname || '').toLowerCase();
    const buf = readFileSync(filePath);

    if (originalName.endsWith('.mxl') || (buf[0] === 0x50 && buf[1] === 0x4B)) {
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

    const VALID_SOLFEGE = new Set(['Do', 'Di', 'Re', 'Ri', 'Me', 'Mi', 'Fa', 'Fi', 'Sol', 'Si', 'La', 'Li', 'Te', 'Ti', '?']);
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
    console.error('[Solfai] MusicXML parse error:', err.message);
    return res.status(500).json({ error: 'Failed to parse MusicXML: ' + err.message });
  } finally {
    try { unlinkSync(filePath); } catch (e) { /* ignore */ }
  }
});

// ─── Manual Note Entry Route ──────────────────────────────
app.post('/api/manual-solfege', (req, res) => {
  const { notes, key } = req.body;
  if (!notes) return res.status(400).json({ error: 'No notes provided' });

  const tonic = (key || 'C').replace(/\s*(major|minor)/i, '').trim();

  const rawMeasures = notes.split('|').map(s => s.trim()).filter(Boolean);
  const measures = rawMeasures.map((m, i) => {
    const noteList = m.split(/[\s,]+/).filter(Boolean);
    return { num: i + 1, notes: noteList, lyrics: '' };
  });

  const VALID_SOLFEGE = new Set(['Do', 'Di', 'Re', 'Ri', 'Me', 'Mi', 'Fa', 'Fi', 'Sol', 'Si', 'La', 'Li', 'Te', 'Ti', '?']);
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
app.listen(PORT, () => console.log(`[Solfai v10] Running on port ${PORT}`));
