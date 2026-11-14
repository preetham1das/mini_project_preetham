export function getDetectionMethodPresentation(result) {
  const method = result?.detection_method;
  if (method === 'temporal') {
    return {
      label: 'Temporal U-Net – Bi-temporal deforestation prediction',
      shortLabel: 'Temporal U-Net',
      description: 'Temporal predictions are retained only where Stage 1 confirms forest-to-non-forest loss.',
    };
  }
  if (method === 'baseline') {
    return {
      label: 'Post-classification – U-Net mask comparison',
      shortLabel: 'Post-classification baseline',
      description: 'Stage 1 forest masks are compared between T1 and T2.',
    };
  }
  if (method === 'compare') {
    return {
      label: 'Model comparison – Temporal U-Net vs Post-classification',
      shortLabel: 'Model comparison',
      description: 'Both methods ran; the response identifies their predictions separately.',
    };
  }
  return {
    label: 'Detection method unavailable',
    shortLabel: 'Method unavailable',
    description: 'The API response did not identify which detection method produced this result.',
  };
}

export function getDetectionWarnings(result) {
  if (Array.isArray(result?.warnings)) return result.warnings.filter((warning) => typeof warning === 'string' && warning.length > 0);
  if (typeof result?.model_note === 'string' && result.model_note.length > 0) return [result.model_note];
  return [];
}

export function getForestChangeSignalPresentation(signal) {
  const normalizedSignal = typeof signal === 'string' ? signal.toUpperCase() : '';
  if (normalizedSignal === 'DEFORESTATION') {
    return { label: 'Deforestation · forest loss', tone: 'loss' };
  }
  if (normalizedSignal === 'NO_DEFORESTATION') {
    return { label: 'No Deforestation Detected', tone: 'neutral' };
  }
  return null;
}
