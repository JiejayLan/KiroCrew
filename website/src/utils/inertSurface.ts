export function isInertSurface(surface: Element | null): boolean {
  return surface !== null && surface.closest('[inert]') !== null
}
