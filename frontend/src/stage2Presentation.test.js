import test from 'node:test';
import assert from 'node:assert/strict';
import { getDetectionMethodPresentation, getDetectionWarnings } from './stage2Presentation.js';

test('method labels are selected from API response metadata', () => {
  assert.match(getDetectionMethodPresentation({ detection_method: 'temporal' }).label, /Temporal U-Net/);
  assert.match(getDetectionMethodPresentation({ detection_method: 'baseline' }).label, /Post-classification/);
  assert.match(getDetectionMethodPresentation({ detection_method: 'compare' }).label, /Model comparison/);
  assert.match(getDetectionMethodPresentation({}).label, /unavailable/i);
});

test('warnings are taken from the response and absent methods do not imply a Stage-1 warning', () => {
  assert.deepEqual(getDetectionWarnings({ detection_method: 'temporal', warnings: ['Temporal domain shift'] }), ['Temporal domain shift']);
  assert.deepEqual(getDetectionWarnings({ detection_method: 'temporal' }), []);
  assert.deepEqual(getDetectionWarnings({ detection_method: 'baseline', warnings: ['Stage-1 domain shift'] }), ['Stage-1 domain shift']);
});
