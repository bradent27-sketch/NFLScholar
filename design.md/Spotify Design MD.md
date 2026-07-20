---
version: alpha
name: Spotify
description: >-
  A music streaming platform with a dark-first, high-contrast design philosophy that prioritizes content discovery and
  playback control through a minimalist interface with strategic accent colors.
logo:
  src: https://open.spotifycdn.com/cdn/images/favicon32.b64ecc03.png
colors:
  surface: '#121212'
  surface-dim: '#0a0a0a'
  surface-bright: '#1e1e1e'
  surface-container-lowest: '#0f0f0f'
  surface-container-low: '#161616'
  surface-container: '#1a1a1a'
  surface-container-high: '#282828'
  surface-container-highest: '#313131'
  on-surface: '#ffffff'
  on-surface-variant: '#b3b3b3'
  inverse-surface: '#ececec'
  inverse-on-surface: '#121212'
  outline: '#535353'
  outline-variant: '#404040'
  surface-tint: '#13863b'
  primary: '#17a147'
  on-primary: '#000000'
  primary-container: '#1aa34a'
  on-primary-container: '#ffffff'
  inverse-primary: '#1ed760'
  secondary: '#b3b3b3'
  on-secondary: '#121212'
  secondary-container: '#282828'
  on-secondary-container: '#ffffff'
  tertiary: '#4687d6'
  on-tertiary: '#ffffff'
  tertiary-container: '#2d5aa8'
  on-tertiary-container: '#e8f0ff'
  error: '#e61e32'
  on-error: '#ffffff'
  error-container: '#8b0a1a'
  on-error-container: '#ffb4ab'
  primary-fixed: '#17a147'
  primary-fixed-dim: '#0f7031'
  on-primary-fixed: '#000000'
  on-primary-fixed-variant: '#0d3d1f'
  secondary-fixed: '#b3b3b3'
  secondary-fixed-dim: '#8f8f8f'
  on-secondary-fixed: '#121212'
  on-secondary-fixed-variant: '#3a3a3a'
  tertiary-fixed: '#4687d6'
  tertiary-fixed-dim: '#3563a8'
  on-tertiary-fixed: '#ffffff'
  on-tertiary-fixed-variant: '#1e4080'
  background: '#121212'
  on-background: '#ffffff'
  surface-variant: '#282828'
typography:
  display:
    fontFamily: SpotifyMixUITitle, SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 56px
    fontWeight: '700'
    lineHeight: 64px
    letterSpacing: '-0.02em'
  headline-lg:
    fontFamily: SpotifyMixUITitle, SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: '-0.01em'
  headline-md:
    fontFamily: SpotifyMixUITitle, SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
  title-lg:
    fontFamily: SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: SpotifyMixUI, CircularSp-Deva, Helvetica Neue, helvetica, arial, sans-serif
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 2px
  DEFAULT: 4px
  md: 6px
  lg: 8px
  xl: 16px
  full: 9999px
spacing:
  unit: 8px
  xs: 4px
  sm: 12px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  container-max: 1248px
elevation:
  sm: 0px 2px 10px -3px rgba(0, 0, 0, 0.4)
  md: 0 3px 8px rgba(0, 0, 0, 0.15)
  lg: 0px 3px 5px -1px rgba(0, 0, 0, 0.2), 0px 6px 10px 0px rgba(0, 0, 0, 0.14), 0px 1px 18px 0px rgba(0, 0, 0, 0.12)
layout:
  containerMaxWidth: 1248px
  gridColumns: 12
components:
  button-primary:
    backgroundColor: '{colors.primary}'
    textColor: '{colors.on-primary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 12px 32px
    height: 48px
    fontWeight: '600'
  button-primary-hover:
    backgroundColor: '{colors.primary-container}'
    textColor: '{colors.on-primary-container}'
  button-primary-active:
    backgroundColor: '#1aa34a'
    textColor: '{colors.on-primary}'
  button-secondary:
    backgroundColor: transparent
    textColor: '{colors.on-surface}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 8px 24px
    height: 40px
    border: 1px solid {colors.on-surface-variant}
  button-secondary-hover:
    backgroundColor: '{colors.surface-container-high}'
    textColor: '{colors.on-surface}'
  button-ghost:
    backgroundColor: transparent
    textColor: '{colors.on-surface-variant}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 8px 16px
    height: 40px
  button-ghost-hover:
    backgroundColor: rgba(255, 255, 255, 0.1)
    textColor: '{colors.on-surface}'
  card:
    backgroundColor: '{colors.surface-container}'
    rounded: '{rounded.md}'
    padding: '{spacing.md}'
    border: none
  card-hover:
    backgroundColor: '{colors.surface-container-high}'
    boxShadow: '{elevation.md}'
  card-elevated:
    backgroundColor: '{colors.surface-container-high}'
    rounded: '{rounded.lg}'
    padding: '{spacing.md}'
    boxShadow: '{elevation.lg}'
  input-field:
    backgroundColor: '{colors.surface-container-low}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    rounded: '{rounded.DEFAULT}'
    padding: 12px 16px
    border: 1px solid {colors.outline-variant}
  input-field-focus:
    backgroundColor: '{colors.surface-container}'
    borderColor: '{colors.primary}'
    boxShadow: 0 0 0 2px rgba(30, 215, 96, 0.2)
  chip:
    backgroundColor: '{colors.surface-container-high}'
    textColor: '{colors.on-surface}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 8px 16px
    height: 32px
  chip-active:
    backgroundColor: '{colors.primary}'
    textColor: '{colors.on-primary}'
  badge:
    backgroundColor: '{colors.tertiary-container}'
    textColor: '{colors.on-tertiary-container}'
    typography: '{typography.label-sm}'
    rounded: '{rounded.full}'
    padding: 4px 12px
  list-item:
    backgroundColor: transparent
    rounded: '{rounded.md}'
    padding: '{spacing.sm}'
    textColor: '{colors.on-surface}'
  list-item-hover:
    backgroundColor: '{colors.surface-container-high}'
    textColor: '{colors.on-surface}'
---

## Overview

Spotify's design system embodies a "Dark-First Minimalism" aesthetic—a deliberate inversion of traditional UI hierarchies where deep blacks (#121212, #0a0a0a) form the canvas, and a vibrant neon green (#1ed760) serves as the singular accent that commands attention. The brand personality is confident, energetic, and culturally attuned: the interface recedes to let content (album art, artist photos, playlists) take center stage, while interaction elements appear only when needed. The emotional response is one of clarity and focus—users feel immersed in music, not distracted by chrome. Voice: conversational yet authoritative, never apologetic. Example: "Your Library keeps your favorite songs, podcasts, and playlists in one place—always ready to play."

## Colors

The color system operates on a principle of maximum contrast and intentional restraint. Primary (#1ed760) is Spotify's signature green, deployed exclusively on call-to-action buttons (48px height, 12px vertical padding), active states, and focus indicators—it appears sparingly but unmistakably. The surface stack descends from #121212 (base) through #282828 (container-high) to #0a0a0a (dim), creating subtle depth without sacrificing legibility. Secondary (#b3b3b3) provides mid-tone text for supporting information and disabled states. Tertiary (#4687d6) reserves a cool blue for tooltips and secondary actions, appearing in the CSS variable --generic-tooltip-background-color. Error (#e61e32) signals destructive actions with urgency. On-surface text is pure white (#ffffff) for maximum contrast

## Typography

The type system prioritizes legibility and hierarchy through weight and size rather than color variation. SpotifyMixUI (Spotify's proprietary typeface) and CircularSp variants form the primary stack, with Helvetica Neue as fallback. Display (56px, 700 weight, -0.02em tracking) anchors hero sections; headline-lg (32px, 700 weight) titles major sections like "Trending songs" and "Popular artists"; body-md (16px, 400 weight, 24px line-height) carries all primary content. Label-md (14px, 600 weight, 0.01em tracking) is used for button text and metadata tags. The system avoids color-based emphasis—instead, weight and size do the work. On small labels over busy album artwork, apply text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) to maintain readability without adding visual noise.

## Layout

The layout follows a 12-column grid with a max-width of 1248px, centering content on larger screens and flowing edge-to-edge on mobile. The left sidebar (Your Library, navigation) is fixed at ~300px width on desktop, collapsible on tablet. Main content area uses 24px gutters (md spacing) between sections; vertical rhythm is maintained with 40px (lg spacing) between major sections like "Trending songs" and "Popular artists." Cards and tiles use 12px (sm spacing) internal padding for compact density. The design favors whitespace asymmetrically—more breathing room above headlines, tighter spacing within content grids—to guide the eye downward and rightward. Container max-width prevents content from stretching excessively on ultra-wide displays, preserving focus.

## Elevation & Depth

Depth is conveyed through subtle shadow and background color shifts rather than aggressive drop shadows. Level 1 (Base): surface (#121212) with no shadow, used for the main canvas. Level 2 (Standard Card): surface-container (#1a1a1a) with box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15), applied to cards and modals on hover. Level 3 (Elevated): surface-container-high (#282828) with box-shadow: 0px 3px 5px -1px rgba(0, 0, 0, 0.2), 0px 6px 10px 0px rgba(0, 0, 0, 0.14), 0px 1px 18px 0px rgba(0, 0, 0, 0.12), reserved for floating panels and notifications. The shadow palette uses only black with varying o

## Shapes

The shape philosophy is "Functional Geometry"—rounded corners are applied strategically based on component type and interaction frequency. Buttons use full radius (9999px) to signal interactivity and approachability; their 48px height and 12px padding create a comfortable tap target. Cards and containers use 6px (md) or 8px (lg) radius for a modern, slightly softened look that avoids the sterility of sharp corners. Input fields use 4px (DEFAULT) radius for a more technical, form-like appearance. The 50% radius (observed on artist avatars and circular badges) is reserved for profile images and

## Components

### Action Elements
Buttons are the primary interaction primitive. Primary buttons (button-primary) use #1ed760 background, black text, 48px height, 12px vertical + 32px horizontal padding, and full border-radius (9999px). On hover (button-primary-hover), the background shifts to #1aa34a (primary-container) with a 200ms ease transition. Secondary buttons (button-secondary) are transparent with a 1px border in on-surface-variant (#b3b3b3), same dimensions, and white text; on hover, the background becomes surface-container-high (#282828). Ghost buttons (button-ghost) have no border or background, only text in on-surface-variant; on hover, a subtle rgba(255, 255, 255, 0.1) background appears. All buttons use label-md typography (14px, 600 weight) and disable at opacity: 0.5 with cursor: not-a

## Do's and Don'ts

**Do**
- Do use primary green (#1ed760) exclusively for CTAs, active states, and focus indicators—never for body text or secondary UI.
- Do maintain high contrast: white text on dark surfaces, dark text on green surfaces (minimum 7:1 WCAG AA ratio).
- Do apply 24px (md) gutters between major sections and 12px (sm) padding within cards to create visual rhythm.
- Do use full border-radius (9999px) on buttons and chips to signal interactivity; reserve 4–8px for containers.
- Do transition background colors and shadows over 150–200ms ease on hover; avoid instant state changes.
- Do layer shadows using only black with opacity (0.12–0.2) to preserve the dark theme's color integrity.

**Don't**
- Don't use primary green for non-interactive elements or as a background color—it must remain a focal accent.
- Don't apply text-shadow or glow effects to body text; reserve shadows for small labels over busy imagery only.
- Don't exceed 1248px container max-width; let content breathe with asymmetric whitespace rather than stretching.
- Don't use rounded corners smaller than 2px or larger than 16px outside of full-radius buttons; maintain the geometric hierarchy.
- Don't animate opacity or transform on every hover; use color and shadow transitions to keep the interface responsive and lightweight.
- Don't mix surface colors arbitrarily; always reference the surface stack (dim → base → container-low/high) to maintain depth consistency.
