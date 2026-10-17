import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ConfirmDialog } from './ConfirmDialog'

describe('ConfirmDialog Escape handling', () => {
  it('stands down while inert and leaves Escape for a later capture listener', () => {
    const onCancel = vi.fn()
    render(
      <ConfirmDialog
        title="Delete note"
        body="Move this note to trash?"
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={() => {}}
        onCancel={onCancel}
      />,
    )
    screen.getByRole('dialog').setAttribute('inert', '')
    const scannerEscape = vi.fn()
    document.addEventListener('keydown', scannerEscape, { capture: true })
    try {
      fireEvent.keyDown(document.body, { key: 'Escape' })
      expect(onCancel).not.toHaveBeenCalled()
      expect(scannerEscape).toHaveBeenCalledTimes(1)
    } finally {
      document.removeEventListener('keydown', scannerEscape, { capture: true })
    }
  })
})
