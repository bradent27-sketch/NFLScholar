---
version: alpha
name: Dunks & Threes
description: >-
  NBA stats, analysis, and predictions platform featuring advanced metrics like Estimated Plus-Minus (EPM). A
  data-driven interface for basketball enthusiasts and analysts.
logo:
  src: https://dunksandthrees.com/favicon.svg
colors:
  surface: '#1a1a1a'
  surface-dim: '#0f0f0f'
  surface-bright: '#2a2a2a'
  surface-container-lowest: '#0a0a0a'
  surface-container-low: '#141414'
  surface-container: '#1a1a1a'
  surface-container-high: '#242424'
  surface-container-highest: '#2e2e2e'
  on-surface: '#f8f8f8'
  on-surface-variant: '#b3b3b3'
  inverse-surface: '#f0f0f0'
  inverse-on-surface: '#1a1a1a'
  outline: '#666666'
  outline-variant: '#4a4a4a'
  surface-tint: '#03aa7d'
  primary: '#03aa7d'
  on-primary: '#f8f8f8'
  primary-container: '#027a5a'
  on-primary-container: '#e8f8f5'
  inverse-primary: '#047857'
  secondary: '#feda6a'
  on-secondary: '#1a1a1a'
  secondary-container: '#b8a84a'
  on-secondary-container: '#0f0f0f'
  tertiary: '#c0c0c7'
  on-tertiary: '#1a1a1a'
  tertiary-container: '#757575'
  on-tertiary-container: '#f0f0f0'
  error: '#ef4444'
  on-error: '#f8f8f8'
  error-container: '#b42e28'
  on-error-container: '#fce8e8'
  primary-fixed: '#047857'
  primary-fixed-dim: '#03aa7d'
  on-primary-fixed: '#f0f9f7'
  on-primary-fixed-variant: '#027a5a'
  secondary-fixed: '#feda6a'
  secondary-fixed-dim: '#d4b84a'
  on-secondary-fixed: '#2a2a1a'
  on-secondary-fixed-variant: '#8a7a3a'
  tertiary-fixed: '#c0c0c7'
  tertiary-fixed-dim: '#9a9aa3'
  on-tertiary-fixed: '#0f0f0f'
  on-tertiary-fixed-variant: '#505050'
  background: '#0f0f0f'
  on-background: '#f8f8f8'
  surface-variant: '#2a2a2a'
typography:
  display:
    fontFamily: Nunito Variable
    fontSize: 56px
    fontWeight: '700'
    lineHeight: 64px
    letterSpacing: '-0.02em'
  headline-lg:
    fontFamily: Nunito Variable
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: '-0.015em'
  headline-md:
    fontFamily: Nunito Variable
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: 0em
  title-lg:
    fontFamily: Nunito Variable
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: 0.005em
  body-lg:
    fontFamily: Nunito Variable
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Nunito Variable
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Nunito Variable
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Nunito Variable
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.03em
  mono-md:
    fontFamily: JetBrains Mono Variable
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
rounded:
  sm: 2px
  DEFAULT: 4px
  md: 6px
  lg: 8px
  xl: 12px
  full: 9999px
spacing:
  unit: 8px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  gutter: 16px
  container-max: 1400px
elevation:
  sm: 0 1px 3px rgba(0, 0, 0, 0.12)
  md: 0 3px 8px rgba(0, 0, 0, 0.15)
  lg: 0 8px 16px rgba(0, 0, 0, 0.2)
layout:
  containerMaxWidth: 1400px
  gridColumns: 12
components:
  button-primary:
    backgroundColor: '{colors.primary}'
    textColor: '{colors.on-primary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.sm}'
    padding: 8px 16px
    height: 32px
    border: none
  button-primary-hover:
    backgroundColor: '{colors.primary-container}'
    textColor: '{colors.on-primary}'
    transition: background-color 150ms cubic-bezier(0.4, 0, 0.2, 1)
  button-secondary:
    backgroundColor: transparent
    textColor: '{colors.on-surface-variant}'
    typography: '{typography.label-md}'
    rounded: '{rounded.DEFAULT}'
    padding: 8px 12px
    height: 32px
    border: 1px solid {colors.outline-variant}
  button-secondary-hover:
    backgroundColor: '{colors.surface-container-high}'
    textColor: '{colors.on-surface}'
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1)
  card:
    backgroundColor: '{colors.surface-container}'
    rounded: '{rounded.DEFAULT}'
    padding: '{spacing.md}'
    border: 1px solid {colors.outline-variant}
  card-hover:
    backgroundColor: '{colors.surface-container-high}'
    transition: background-color 200ms cubic-bezier(0.4, 0, 0.2, 1)
  input-field:
    backgroundColor: '{colors.surface-container-low}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    rounded: '{rounded.DEFAULT}'
    padding: 8px 12px
    border: 1px solid {colors.outline-variant}
    height: 36px
  input-field-focus:
    borderColor: '{colors.primary}'
    boxShadow: 0 0 0 2px rgba(3, 170, 125, 0.1)
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1)
  tab-active:
    backgroundColor: transparent
    textColor: '{colors.primary}'
    typography: '{typography.label-md}'
    borderBottom: 2px solid {colors.primary}
    paddingBottom: 8px
  tab-inactive:
    backgroundColor: transparent
    textColor: '{colors.on-surface-variant}'
    typography: '{typography.label-md}'
    borderBottom: 2px solid transparent
    paddingBottom: 8px
  tab-inactive-hover:
    textColor: '{colors.on-surface}'
    transition: color 150ms cubic-bezier(0.4, 0, 0.2, 1)
  badge:
    backgroundColor: '{colors.primary-container}'
    textColor: '{colors.on-primary-container}'
    typography: '{typography.label-sm}'
    rounded: '{rounded.full}'
    padding: 4px 8px
    display: inline-block
  badge-secondary:
    backgroundColor: '{colors.secondary-container}'
    textColor: '{colors.on-secondary-container}'
    typography: '{typography.label-sm}'
    rounded: '{rounded.full}'
    padding: 4px 8px
  table-header:
    backgroundColor: '{colors.surface-container-high}'
    textColor: '{colors.on-surface-variant}'
    typography: '{typography.label-md}'
    padding: 12px 16px
    borderBottom: 1px solid {colors.outline-variant}
  table-row:
    backgroundColor: '{colors.surface-container}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    padding: 12px 16px
    borderBottom: 1px solid {colors.outline-variant}
  table-row-hover:
    backgroundColor: '{colors.surface-container-high}'
    transition: background-color 150ms cubic-bezier(0.4, 0, 0.2, 1)
  stat-positive:
    textColor: '#03aa7d'
    typography: '{typography.label-md}'
    fontWeight: '600'
  stat-negative:
    textColor: '#ef4444'
    typography: '{typography.label-md}'
    fontWeight: '600'
  stat-neutral:
    textColor: '{colors.on-surface-variant}'
    typography: '{typography.label-md}'
    fontWeight: '400'
---

## Overview

Dunks & Threes is a data-intensive basketball analytics platform designed for serious fans, coaches, and analysts who demand precision in their statistical insights. The design system embraces a 'Data-Minimalist' aesthetic: a dark, high-contrast canvas (oklch(22.5% 0 0) background with oklch(98.5% 0 0) text) that prioritizes information density and rapid pattern recognition over decorative flourishes. The interface uses a teal-green primary accent (#03aa7d in dark mode, #047857 in light) paired with strategic use of warm yellows (#feda6a) and reds (#ef4444) for status differentiation—creating a visual language where color carries semantic meaning rather than decoration. The brand personality is analytical yet approachable: confident in its data, never condescending to the user. Example sentence: 'OKC's offensive rating jumped 3.2 points per 100 possessions after the trade deadline—here's why.'

## Colors

The color system is built on a dark-first philosophy optimized for extended viewing of statistical tables and charts. Primary (#03aa7d in dark mode, #047857 in light) is the signature teal accent used exclusively for interactive CTAs, active tab indicators, and positive performance metrics. It appears in the header 'SUBSCRIBE' button and throughout the interface as the focus state for inputs and links. Secondary (#feda6a) is a warm yellow reserved for neutral or secondary data points in charts and badges—it provides visual relief from the cool teal without competing for attention. Tertiary (#c0c0c7) is a cool gray used for tertiary UI elements and chart series. Error (#ef4444) signals negative performance or destructive actions. The surface stack is carefully calibrated: surface-container

## Typography

The typography system uses Nunito Variable as the primary typeface, chosen for its excellent readability at small sizes and its neutral, professional character. The hierarchy is steep and functional: display (56px, 700 weight) is reserved for hero moments; headline-lg (32px, 600 weight) for page titles like 'NBA Team Stats'; headline-md (24px, 600 weight) for section headers; title-lg (18px, 600 weight) for card titles and tab labels; body-lg (16px, 400 weight) for primary content; body-md (14px, 400 weight) for secondary content and table cells; label-md (12px, 500 weight) for buttons, badges, and UI labels; and label-sm (11px, 500 weight) for footnotes and tertiary information. Letter-spacing increases at smaller sizes (0.02em at label-md, 0.03em at label-sm) to maintain legibility in de

## Layout

The layout uses a 12-column grid with a maximum container width of 1400px, accommodating both desktop analytics workstations and tablet-based viewing. The page rhythm is driven by a semantic spacing scale: gutter (16px) for horizontal padding in containers, md (16px) for internal card padding, lg (24px) for section separation, and xl (32px) for major layout breaks. The header is fixed at the top with 40px horizontal padding (sm:px-10 in Tailwind), creating a stable navigation anchor. The main content area uses a two-column layout on desktop (data table on the left, visualization on the right) that collapses to single-column on tablet. White-space is minimal but intentional: 8px gaps between table rows, 12px padding inside table cells, and 16px margins between major sections. This density r

## Elevation & Depth

Depth is achieved through subtle layering and strategic use of borders rather than heavy shadows, maintaining the clean, technical aesthetic. Level 1 (Base): The background (oklch(22.5% 0 0)) is the foundation. Level 2 (Standard Card): Cards use a 1px solid border in outline-variant (#4a4a4a) with a background of surface-container (#1a1a1a) and a subtle box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15). Level 3 (Elevated/Hover): On hover, cards transition to surface-container-high (#242424) with the same border and shadow, creating a 2-step elevation system. Modals and dropdowns use box-shadow: 0 8px

## Shapes

The shape philosophy is 'Technical Precision'—minimal rounding that signals functionality without softness. Buttons and interactive elements use rounded.sm (2px), creating a subtle but distinct corner treatment that feels intentional and data-driven. Cards and containers use rounded.DEFAULT (4px), providing slightly more breathing room while maintaining the technical aesthetic. Modals, dropdowns, and elevated surfaces use rounded.md (6px) to rounded.lg (8px), creating a visual hierarchy where more prominent elements have slightly more generous rounding. Badges and pills use rounded.full (9999p

## Components

### Action Elements
Buttons follow a two-tier system: primary buttons use {colors.primary} (#03aa7d) background with {colors.on-primary} text, 8px vertical / 16px horizontal padding, and rounded.sm (2px) corners. The 'SUBSCRIBE' button in the header exemplifies this: 12px font-size, 500 weight, 32px height. On hover, the background transitions to {colors.primary-container} (#027a5a) with a 150ms cubic-bezier(0.4, 0, 0.2, 1) transition. Secondary buttons use a transparent background with a 1px border in {colors.outline-variant} (#4a4a4a) and {colors.on-surface-variant} text, transitioning to {colors.surface-container-high} on hover. All buttons have a minimum height of 32px and use label-md typography.

### Containers & Surfaces
Cards are the primary container element: {colors.surface-conta

## Do's and Don'ts

**Do**
- Do use {colors.primary} (#03aa7d) exclusively for primary CTAs, active states, and positive metrics—never for backgrounds or secondary UI.
- Do maintain the 2px border-radius on buttons and interactive elements to reinforce the technical, analytical aesthetic.
- Do apply label-md (12px, 500 weight) to all button text and UI labels for consistent visual hierarchy.
- Do use the surface-container stack (surface-container, surface-container-high, surface-container-highest) for layered card and modal backgrounds.
- Do apply box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15) to elevated cards and dropdowns for clear depth separation.
- Do use JetBrains Mono Variable for all numerical data, statistics, and code snippets to ensure monospace alignment.
- Do transition all interactive states (hover, focus, active) over 150ms using cubic-bezier(0.4, 0, 0.2, 1) for responsive but not jarring feedback.

**Don't**
- Don't use rounded corners larger than rounded.lg (8px) on primary interactive elements—the technical aesthetic demands sharp, intentional corners.
- Don't apply shadows to text or use text-shadow on body-lg or larger text; reserve text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) only for label-md and label-sm over busy backgrounds.
- Don't mix primary and secondary colors in a single component—use one accent color per element to maintain visual clarity in data-dense layouts.
- Don't use {colors.secondary} (#feda6a) for primary CTAs or active states; it's reserved for neutral data points and secondary badges.
- Don't apply padding larger than {spacing.md} (16px) inside table cells or data containers—density is a feature, not a bug.
- Don't use the light mode colors (#047857, #b42e28) in dark mode contexts or vice versa; always respect the theme toggle.
- Don't animate transitions longer than 200ms on interactive elements; the platform prioritizes responsiveness and quick feedback.
